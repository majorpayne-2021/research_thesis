"""
components/classify.py

Batch transaction classification component.

Flow:
1) Load model artifacts from GCS (metadata, scaler, RF model)
2) Read features from BigQuery (aml_prod.txn_features_clean)
3) Drop rows with any nulls in feature columns
4) Scale only columns listed in metadata['features']['to_scale']
5) Predict probabilities & classes using threshold (default: metadata['threshold'] = 0.4)
6) Overwrite predictions to BigQuery table (aml_prod.df_txn_pred) with a Melbourne-time batch timestamp (stored as UTC)

Example (local run):
python components/classify.py \
  --project_id extreme-torch-467913-m6 \
  --artifact_base_uri gs://thesis_model/aml_models/v0.1/
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from typing import Optional, List

import numpy as np
import pandas as pd
import pytz
from google.cloud import bigquery, storage
from joblib import load as joblib_load


# --------------------- Helpers ---------------------

def download_gcs_blob(gcs_uri: str, local_path: str, project: Optional[str] = None) -> str:
    """Download a single object from GCS to local_path. Returns local_path."""
    if not gcs_uri.startswith("gs://"):
        raise ValueError(f"Invalid GCS URI: {gcs_uri}")
    bucket_name, *path_parts = gcs_uri.replace("gs://", "").split("/")
    blob_path = "/".join(path_parts)
    client = storage.Client(project=project)
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    local_dir = os.path.dirname(local_path)
    if local_dir:
        os.makedirs(local_dir, exist_ok=True)
    blob.download_to_filename(local_path)
    return local_path


def read_bq_features(
    project_id: str,
    dataset_id: str,
    table_id: str,
    feature_order: List[str],
) -> pd.DataFrame:
    """Read txId, time_step, and the requested feature columns (already normalized field names)."""
    client = bigquery.Client(project=project_id)
    table_fqn = f"`{project_id}.{dataset_id}.{table_id}`"
    select_feats = ", ".join([f"`{c}`" for c in feature_order])
    query = f"""
    SELECT
      CAST(txId AS STRING) AS txId,
      time_step,
      {select_feats}
    FROM {table_fqn}
    """
    try:
        return client.query(query).result().to_dataframe(create_bqstorage_client=True)
    except Exception:
        # Fallback if bigquery-storage extras are not installed
        return client.query(query).result().to_dataframe()


def write_bq_overwrite(df: pd.DataFrame, project_id: str, dataset_id: str, table_id: str) -> None:
    """Overwrite (truncate) the target BigQuery table with df."""
    client = bigquery.Client(project=project_id)
    table_ref = f"{project_id}.{dataset_id}.{table_id}"
    job_config = bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")
    job = client.load_table_from_dataframe(df, table_ref, job_config=job_config)
    job.result()


def build_feature_matrix(df_feat: pd.DataFrame, meta: dict, scaler) -> pd.DataFrame:
    """Return a DataFrame in the exact training order; scale only the 'to_scale' subset."""
    feat_meta = meta.get("features", {})
    order = list(feat_meta.get("features_order", []))
    to_scale = list(feat_meta.get("to_scale", []))
    already_norm = list(feat_meta.get("already_normalised", []))

    if not order:
        raise ValueError("Model metadata missing features.features_order.")

    # Validate required columns & no overlap
    missing = [c for c in order if c not in df_feat.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")
    overlap = set(to_scale).intersection(set(already_norm))
    if overlap:
        raise ValueError(f"Columns in BOTH to_scale and already_normalised: {overlap}")

    X_df = df_feat[order].copy()
    for c in order:
        X_df[c] = pd.to_numeric(X_df[c], errors="coerce")

    if to_scale:
        scaled_vals = scaler.transform(X_df[to_scale])
        X_df.loc[:, to_scale] = pd.DataFrame(scaled_vals, columns=to_scale, index=X_df.index)

    return X_df


# ---------------------- Main ----------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Run transaction classification model.")
    parser.add_argument("--project_id", required=True)
    parser.add_argument("--input_dataset", default="aml_prod")
    parser.add_argument("--input_table", default="txn_features_clean")
    parser.add_argument("--output_dataset", default="aml_prod")
    parser.add_argument("--output_table", default="df_txn_pred")
    parser.add_argument("--artifact_base_uri", required=True)
    parser.add_argument("--threshold", type=float, default=None)
    args = parser.parse_args()

    project_id = args.project_id
    input_dataset = args.input_dataset
    input_table = args.input_table
    output_dataset = args.output_dataset
    output_table = args.output_table
    base_uri = args.artifact_base_uri.rstrip("/") + "/"

    # --- Load artifacts ---
    local_dir = "/tmp/aml_artifacts"
    meta_path   = download_gcs_blob(base_uri + "model_metadata.json", f"{local_dir}/model_metadata.json", project=project_id)
    model_path  = download_gcs_blob(base_uri + "rf_model.joblib",      f"{local_dir}/rf_model.joblib",      project=project_id)
    scaler_path = download_gcs_blob(base_uri + "scaler.joblib",        f"{local_dir}/scaler.joblib",        project=project_id)

    with open(meta_path, "r") as f:
        meta = json.load(f)

    threshold = args.threshold if args.threshold is not None else float(meta.get("threshold", 0.4))
    model_name = meta.get("model_name", "rf_txn_classifier")
    feat_order = meta.get("features", {}).get("features_order", [])
    if not feat_order:
        raise ValueError("Model metadata missing features.features_order.")

    print(f"[meta] model={model_name} | threshold={threshold:.3f} | features={len(feat_order)}")

    # --- Read features ---
    df_all = read_bq_features(project_id, input_dataset, input_table, feat_order)
    print(f"[read] rows={len(df_all):,} | cols={df_all.shape[1]}")

    # --- Drop rows with null feature values ---
    before = len(df_all)
    df_all = df_all.dropna(subset=feat_order)
    after = len(df_all)
    if before != after:
        print(f"[warn] Dropped {before - after} rows with null feature values ({after} remain).")
    if df_all.empty:
        print("[warn] All rows dropped after null check; exiting.")
        sys.exit(0)

    # --- Build features ---
    scaler = joblib_load(scaler_path)
    X = build_feature_matrix(df_all[feat_order], meta, scaler)

    # --- Predict ---
    model = joblib_load(model_path)
    if not hasattr(model, "predict_proba"):
        raise TypeError("Loaded model does not support predict_proba().")

    proba = model.predict_proba(X)[:, 1]
    label = (proba >= threshold).astype(int)
    label_text = np.where(label == 1, "Illicit", "Licit")

    # --- Melbourne timestamp (stored as UTC in BQ TIMESTAMP) ---
    mel_tz = pytz.timezone("Australia/Melbourne")
    run_ts_local = datetime.now(mel_tz)
    run_ts_utc = pd.Timestamp(run_ts_local.astimezone(pytz.utc))

    # --- Prepare output ---
    out_df = pd.DataFrame({
        "txId": df_all["txId"].astype(str),
        "time_step": pd.to_numeric(df_all["time_step"], errors="coerce").astype("Int64"),
        "pred_model": model_name,
        "pred_model_threshold": float(threshold),
        "pred_proba": proba.astype(float),
        "pred_class": label.astype(int),
        "pred_class_label": label_text.astype(object),
        "run_timestamp": run_ts_utc,
    })

    print(f"[info] Melbourne batch timestamp: {run_ts_local.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    print(f"[write] OVERWRITING {project_id}.{output_dataset}.{output_table} with {len(out_df):,} rows")

    # --- Overwrite target table instead of append ---
    write_bq_overwrite(out_df, project_id, output_dataset, output_table)
    print("[done] Classification complete.")


if __name__ == "__main__":
    main()
