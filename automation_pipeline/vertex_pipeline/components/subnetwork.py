#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Subnetwork builder component for Vertex AI Pipeline.

This script:
  1. Reads transaction edges and predicted labels from BigQuery
  2. Uses local txn_subnetworks.py module to build illicit subnetworks
  3. Writes resulting edges and nodes tables back to BigQuery

Assumes txn_subnetworks.py is located in the same folder.
"""

import argparse
import logging
import os
import sys
import pandas as pd
from google.cloud import bigquery
import importlib.util

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logging.info("ARGV: %s", sys.argv)
logging.info("CWD: %s | LS: %s", os.getcwd(), ", ".join(os.listdir(".")))


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

# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------
def bq_client(project_id: str) -> bigquery.Client:
    return bigquery.Client(project=project_id)


def bq_read(project_id: str, sql: str) -> pd.DataFrame:
    client = bq_client(project_id)
    return client.query(sql).result().to_dataframe(create_bqstorage_client=True)


def bq_write(project_id: str, df: pd.DataFrame, dataset: str, table: str) -> None:
    client = bq_client(project_id)
    dest = f"{project_id}.{dataset}.{table}"
    job = client.load_table_from_dataframe(
        df, dest, job_config=bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")
    )
    job.result()
    logging.info(f"Wrote {len(df):,} rows to {dest}")


# -----------------------------------------------------------------------------
# Main workflow
# -----------------------------------------------------------------------------
def run(args):
    logging.info(
    "CONFIG project_id=%s | edges=%s.%s | preds=%s.%s | outputs=%s (edges_out=%s, nodes_out=%s) | dry_run=%s",
    args.project_id, args.edgeset_dataset, args.edgeset_table,
    args.pred_dataset, args.pred_table,
    args.output_dataset, args.edges_output_table, args.nodes_output_table,
    args.dry_run,)

    logging.info("Reading edge list...")
    edges_sql = f"SELECT txId1, txId2 FROM `{args.project_id}.{args.edgeset_dataset}.{args.edgeset_table}`"
    df_edges = bq_read(args.project_id, edges_sql)

    logging.info("Reading predictions...")
    pred_sql = f"SELECT txId, pred_class_label FROM `{args.project_id}.{args.pred_dataset}.{args.pred_table}`"
    df_pred = bq_read(args.project_id, pred_sql)

    illicit_value = "Illicit"
    seed_txns = df_pred.loc[df_pred["pred_class_label"] == illicit_value, "txId"].tolist()
    logging.info(f"Illicit seeds: {len(seed_txns):,}")

    if not seed_txns:
        logging.warning("No illicit seeds found — exiting.")
        return

    logging.info("Building subnetworks...")
    nodes_all, edges_all = build_network.build_subnetworks_naive(
        edges_df=df_edges,
        labels_df=df_pred,
        seed_txns=seed_txns,
        illicit_value=illicit_value,
        src_col="txId1",
        dst_col="txId2",
        label_id_col="txId",
        label_col="pred_class_label",
        progress=True,
        progress_every=args.progress_every,
    )

    logging.info("Deduplicating subnetworks...")
    nodes_dedup, edges_dedup, _, _ = build_network.deduplicate_subnetworks_by_node_subset(
        nodes_all, edges_all, relabel=False, progress=True
    )

    logging.info("Merging subnetworks...")
    _, _, txn_final, edges_final = build_network.merge_subnetworks_by_node_overlap(
        nodes_dedup,
        edges_dedup,
        min_shared_nodes=args.min_shared_nodes,
        progress=True,
        print_unmerged=False,
        collapse=True,
    )

    logging.info(f"Final nodes: {len(txn_final):,} | edges: {len(edges_final):,}")

    if not args.dry_run:
        bq_write(args.project_id, edges_final, args.output_dataset, args.edges_output_table)
        bq_write(args.project_id, txn_final, args.output_dataset, args.nodes_output_table)
    else:
        logging.info("Dry-run mode: results not written to BigQuery.")


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Build illicit subnetworks.")
    parser.add_argument("--project_id", required=True)
    parser.add_argument("--edgeset_dataset", default="txn")
    parser.add_argument("--edgeset_table", default="txn_edgelist")
    parser.add_argument("--pred_dataset", default="aml_prod")
    parser.add_argument("--pred_table", default="df_txn_pred")
    parser.add_argument("--output_dataset", default="aml_prod")
    parser.add_argument("--edges_output_table", default="df_network_edges")
    parser.add_argument("--nodes_output_table", default="df_nw_txn_final")
    parser.add_argument("--min_shared_nodes", type=int, default=1)
    parser.add_argument("--progress_every", type=int, default=100)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    run(args)


if __name__ == "__main__":
    main()

