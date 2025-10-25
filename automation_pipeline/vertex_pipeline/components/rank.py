"""
components/rank.py

Builds per-subnetwork PageRank and a composite ranking using txn_rank.py.

Defaults match your notebook edits:
- Inputs:
    * Features: aml_prod.txn_features_clean  (txId, in_BTC_total, in_txs_degree, out_txs_degree)
    * Edges:    aml_prod.df_network_edges    (merged_subnetwork_id, src_txn_id, dst_txn_id, min_src_hop, min_dst_hop)
    * Nodes:    aml_prod.df_nw_txn_final     (merged_subnetwork_id, txn_id, min_hop)
- Module: gs://thesis_classes/txn_rank.py  (local fallback: /mnt/data/txn_rank.py)
- PageRank params: alpha=0.85, max_iter=100, tol=1e-8
- Composite weights: {'pr':0.60,'val':0.30,'in':0.07,'out':0.03}
- Output: overwrites aml_prod.df_rank_composite when --write_to_bq is passed

Usage example:
python components/rank.py \
  --project_id extreme-torch-467913-m6 \
  --write_to_bq
"""

import argparse
import importlib.util
import numpy as np
import pandas as pd
from google.cloud import bigquery, storage
import sys, os, logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logging.info("ARGV: %s", sys.argv)
logging.info("CWD: %s | LS: %s", os.getcwd(), ", ".join(os.listdir(".")))

# -----------------------------------------------------------------------------
# Dynamically load txn_rank.py
# -----------------------------------------------------------------------------

import importlib.util
import os

MODULE_PATH = os.path.join(os.path.dirname(__file__), "txn_rank.py")
spec = importlib.util.spec_from_file_location("txn_rank", MODULE_PATH)
txn_rank = importlib.util.module_from_spec(spec)
spec.loader.exec_module(txn_rank)

# Instantiate class from the loaded module
build_rank = txn_rank.build_txn_rank()


# --------------------- Helpers ---------------------

def download_gcs(gcs_uri: str, local_path: str, project: str | None = None) -> str:
    if not gcs_uri.startswith("gs://"):
        raise ValueError(f"Invalid GCS URI: {gcs_uri}")
    bucket, *parts = gcs_uri.replace("gs://","").split("/")
    storage.Client(project=project).bucket(bucket).blob("/".join(parts)).download_to_filename(local_path)
    return local_path

def bq_read_sql(project_id: str, sql: str, params=None) -> pd.DataFrame:
    client = bigquery.Client(project=project_id)
    job_cfg = bigquery.QueryJobConfig(query_parameters=params) if params else None
    return client.query(sql, job_config=job_cfg).result().to_dataframe(create_bqstorage_client=True)

def bq_write_df(project_id: str, df: pd.DataFrame, dataset: str, table: str, overwrite: bool = True):
    client = bigquery.Client(project=project_id)
    table_fqn = f"{project_id}.{dataset}.{table}"
    job_config = bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE" if overwrite else "WRITE_APPEND")
    job = client.load_table_from_dataframe(df, table_fqn, job_config=job_config)
    job.result()
    return table_fqn


# ----------------------- Main -----------------------

def main():
    parser = argparse.ArgumentParser(description="Rank subnetworks using PageRank + composite score.")
    parser.add_argument("--project_id", required=True)

    # Inputs
    parser.add_argument("--feats_dataset", default="aml_prod")
    parser.add_argument("--feats_table",   default="txn_features_clean")
    parser.add_argument("--edges_dataset", default="aml_prod")
    parser.add_argument("--edges_table",   default="df_network_edges")
    parser.add_argument("--nodes_dataset", default="aml_prod")
    parser.add_argument("--nodes_table",   default="df_nw_txn_final")

    # Module
    parser.add_argument("--module_gcs_uri",        default="gs://thesis_classes/txn_rank.py")
    parser.add_argument("--module_local_fallback", default="/mnt/data/txn_rank.py")

    # PageRank params
    parser.add_argument("--pr_alpha",   type=float, default=0.85)
    parser.add_argument("--pr_maxiter", type=int,   default=100)
    parser.add_argument("--pr_tol",     type=float, default=1e-8)

    # Composite weights
    parser.add_argument("--w_pr",  type=float, default=0.60)
    parser.add_argument("--w_val", type=float, default=0.30)
    parser.add_argument("--w_in",  type=float, default=0.07)
    parser.add_argument("--w_out", type=float, default=0.03)
    parser.add_argument("--invert_out_deg", action="store_true", default=True)

    # Outputs
    parser.add_argument("--output_dataset",         default="aml_prod")
    parser.add_argument("--pr_output_table",        default="df_rank_pr")
    parser.add_argument("--composite_output_table", default="df_rank_composite")
    parser.add_argument("--write_to_bq", action="store_true", help="Overwrite output tables in BigQuery")

    # Optionally also write the PR table
    parser.add_argument("--write_pr_table", action="store_true", help="Also write df_rank_pr")

    args = parser.parse_args()

    PROJECT_ID = args.project_id

    # ---- Read inputs ----
    df_txn_features = bq_read_sql(
        PROJECT_ID,
        f"""
        SELECT
          CAST(txId AS STRING) AS txId,
          CAST(in_BTC_total AS FLOAT64) AS in_BTC_total,
          CAST(in_txs_degree AS INT64)  AS in_txs_degree,
          CAST(out_txs_degree AS INT64) AS out_txs_degree
        FROM `{PROJECT_ID}.{args.feats_dataset}.{args.feats_table}`
        """
    )
    print(f"[read] features rows: {len(df_txn_features):,}")

    df_network_edges = bq_read_sql(
        PROJECT_ID,
        f"""
        SELECT
          merged_subnetwork_id,
          CAST(src_txn_id AS STRING) AS src_txn_id,
          CAST(dst_txn_id AS STRING) AS dst_txn_id,
          CAST(min_src_hop AS INT64) AS min_src_hop,
          CAST(min_dst_hop AS INT64) AS min_dst_hop
        FROM `{PROJECT_ID}.{args.edges_dataset}.{args.edges_table}`
        """
    )
    print(f"[read] network edges rows: {len(df_network_edges):,}")

    df_nw_txn_final = bq_read_sql(
        PROJECT_ID,
        f"""
        SELECT
          merged_subnetwork_id,
          CAST(txn_id AS STRING) AS txn_id,
          CAST(min_hop AS INT64)  AS min_hop
        FROM `{PROJECT_ID}.{args.nodes_dataset}.{args.nodes_table}`
        """
    )
    print(f"[read] network nodes rows: {len(df_nw_txn_final):,}")

    # ---- PageRank per subnetwork ----
    df_pr = build_rank.build_pagerank(
        df_edges=df_network_edges,
        group='merged_subnetwork_id',
        src='src_txn_id',
        dst='dst_txn_id',
        weight_col=None,
        alpha=args.pr_alpha,
        max_iter=args.pr_maxiter,
        tol=args.pr_tol
    )
    print(f"[rank] pagerank rows: {len(df_pr):,}")

    # ---- Ensure PR exists for ALL nodes (fix single-node subnetworks) ----
    # Normalize IDs
    df_pr["node"] = df_pr["node"].astype(str)
    df_nw_txn_final["txn_id"] = df_nw_txn_final["txn_id"].astype(str)

    # All nodes by (group, node)
    nodes = df_nw_txn_final[["merged_subnetwork_id", "txn_id"]].rename(columns={"txn_id": "node"})
    sizes = nodes.groupby("merged_subnetwork_id")["node"].size().rename("n").reset_index()

    pr_full = (
        nodes.merge(df_pr, how="left", on=["merged_subnetwork_id", "node"])
             .merge(sizes, how="left", on="merged_subnetwork_id")
    )

    pr_full["n"] = pr_full["n"].astype(float)
    mask = pr_full["pagerank"].isna()
    pr_full.loc[mask, "pagerank"] = np.where(
        pr_full.loc[mask, "n"] == 1.0, 1.0, 1.0 / pr_full.loc[mask, "n"]
    )

    df_pr_fixed = pr_full[["merged_subnetwork_id", "node", "pagerank"]].copy()

    # ---- Composite calculation ----
    weights = {"pr": args.w_pr, "val": args.w_val, "in": args.w_in, "out": args.w_out}

    # Normalize feature/PR IDs to strings for safe joins
    df_txn_features["txId"] = df_txn_features["txId"].astype(str)
    df_pr_fixed["node"] = df_pr_fixed["node"].astype(str)

    df_composite, used_weights = build_rank.build_composite_calc(
        df_txn_features=df_txn_features,
        df_pr=df_pr_fixed,
        df_nw_txn_final=df_nw_txn_final,
        nw_txn_col='txn_id',
        group_col='merged_subnetwork_id',
        feats_txid_col='txId',
        pr_node_col='node',
        pr_value_col='pagerank',
        weights=weights,
        invert_out_deg=bool(args.invert_out_deg)
    )
    print("[rank] composite built. weights used:", used_weights)
    print(f"[rank] composite rows: {len(df_composite):,}")

    # ---- Write outputs (overwrite) ----
    if args.write_to_bq:
        if args.write_pr_table:
            pr_fqn = bq_write_df(PROJECT_ID, df_pr_fixed, args.output_dataset, args.pr_output_table, overwrite=True)
            print(f"[write] Overwrote {pr_fqn} with {len(df_pr_fixed):,} rows")

        comp_fqn = bq_write_df(PROJECT_ID, df_composite, args.output_dataset, args.composite_output_table, overwrite=True)
        print(f"[write] Overwrote {comp_fqn} with {len(df_composite):,} rows")
    else:
        print("[dry-run] Skipped writing to BigQuery (use --write_to_bq to enable).")


if __name__ == "__main__":
    main()
