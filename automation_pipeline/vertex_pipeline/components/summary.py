#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
components/summary.py

Builds a subnetwork summary table using txn_subnetworks.reporting.build_subnetwork_viz_summary.

Defaults:
- Rank input:   aml_prod.df_rank_composite
- Actor inputs: actor.addrtxn_edgelist, actor.txaddr_edgelist
- Output:       aml_prod.df_subnetwork_summary (WRITE_TRUNCATE when --write_to_bq)

Usage:
python components/summary.py \
  --project_id extreme-torch-467913-m6 \
  --write_to_bq
"""

import argparse
import importlib.util
import os

import pandas as pd
from google.cloud import bigquery


# ----------------------------- BigQuery helpers -----------------------------

def bq_read_sql(project_id: str, sql: str, params=None) -> pd.DataFrame:
    client = bigquery.Client(project=project_id)
    job_cfg = bigquery.QueryJobConfig(query_parameters=params) if params else None
    return client.query(sql, job_config=job_cfg).result().to_dataframe(create_bqstorage_client=True)


def bq_write_df(project_id: str, df: pd.DataFrame, dataset: str, table: str, overwrite: bool = True):
    client = bigquery.Client(project=project_id)
    table_fqn = f"{project_id}.{dataset}.{table}"
    job_cfg = bigquery.LoadJobConfig(
        write_disposition="WRITE_TRUNCATE" if overwrite else "WRITE_APPEND"
    )
    job = client.load_table_from_dataframe(df, table_fqn, job_config=job_cfg)
    job.result()
    return table_fqn


# --------------------------------- Main ---------------------------------

def main():
    parser = argparse.ArgumentParser(description="Build subnetwork summary table.")
    parser.add_argument("--project_id", required=True)

    # Inputs
    parser.add_argument("--rank_dataset", default="aml_prod")
    parser.add_argument("--rank_table",   default="df_rank_composite")

    parser.add_argument("--actor_dataset", default="actor")
    parser.add_argument("--addrtxn_table", default="addrtxn_edgelist")   # address -> txn
    parser.add_argument("--txaddr_table",  default="txaddr_edgelist")    # txn -> address

    # Output
    parser.add_argument("--output_dataset", default="aml_prod")
    parser.add_argument("--summary_table",  default="df_subnetwork_summary")
    parser.add_argument("--write_to_bq", action="store_true", help="Overwrite output in BigQuery")

    args = parser.parse_args()
    PROJECT_ID = args.project_id

    # -----------------------------------------------------------------------------
    # Dynamically load txn_subnetworks.py
    # -----------------------------------------------------------------------------
    MODULE_PATH = os.path.join(os.path.dirname(__file__), "txn_subnetworks.py")
    spec = importlib.util.spec_from_file_location("txn_subnetworks", MODULE_PATH)
    txn_subnetworks = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(txn_subnetworks)

    # Instantiate classes from the loaded module
    build_network = txn_subnetworks.build_txn_subnetwork()
    build_report  = txn_subnetworks.reporting()
    print(f"[module] using: {MODULE_PATH}")

    # ---- Read inputs ----
    df_txn_rank = bq_read_sql(
        PROJECT_ID,
        f"SELECT * FROM `{PROJECT_ID}.{args.rank_dataset}.{args.rank_table}`"
    )
    print(f"[read] rank rows: {len(df_txn_rank):,}")

    df_addrtxn = bq_read_sql(
        PROJECT_ID,
        f"SELECT * FROM `{PROJECT_ID}.{args.actor_dataset}.{args.addrtxn_table}`"
    )
    print(f"[read] addrtxn rows: {len(df_addrtxn):,}")

    df_txnaddr = bq_read_sql(
        PROJECT_ID,
        f"SELECT * FROM `{PROJECT_ID}.{args.actor_dataset}.{args.txaddr_table}`"
    )
    print(f"[read] txaddr rows: {len(df_txnaddr):,}")

    # ---- Minimal validations / dtype normalization for IDs ----
    need_cols = {"merged_subnetwork_id", "node"}  # required by the module
    missing = need_cols - set(df_txn_rank.columns)
    if missing:
        raise ValueError(
            f"Ranking table missing required columns: {missing}. "
            f"Columns present: {list(df_txn_rank.columns)}"
        )

    # Convert common ID fields to string to avoid join/type mismatches inside the module
    for c in ("node", "txId", "txn_id", "src_txn_id", "dst_txn_id", "address", "addr_id"):
        if c in df_txn_rank.columns:
            df_txn_rank[c] = df_txn_rank[c].astype(str)
        if c in df_addrtxn.columns:
            df_addrtxn[c] = df_addrtxn[c].astype(str)
        if c in df_txnaddr.columns:
            df_txnaddr[c] = df_txnaddr[c].astype(str)

    # ---- Build summary ----
    summary_table = build_report.build_subnetwork_viz_summary(
        df_rank=df_txn_rank,    # expected param name in your module
        df_addrtxn=df_addrtxn,
        df_txnaddr=df_txnaddr,
        join_to_string=True     # keep list-like outputs as strings for easy SQL display
    )
    print(f"[summary] rows: {len(summary_table):,}")

    # ---- Write output ----
    if args.write_to_bq:
        out_fqn = bq_write_df(PROJECT_ID, summary_table, args.output_dataset, args.summary_table, overwrite=True)
        print(f"[write] Overwrote {out_fqn} with {len(summary_table):,} rows")
    else:
        print("[dry-run] Skipped writing to BigQuery (use --write_to_bq to enable).")


if __name__ == "__main__":
    main()
