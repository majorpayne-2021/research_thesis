from kfp import dsl, compiler
from google.cloud import aiplatform

PROJECT_ID = "extreme-torch-467913-m6"
REGION = "us-central1"
IMAGE_URI = "us-central1-docker.pkg.dev/extreme-torch-467913-m6/aml-pipeline/vertex-aml:v0.3"
PIPELINE_ROOT = "gs://thesis_model/aml_models/v0.1/"

@dsl.container_component
def diag_comp():
    return dsl.ContainerSpec(
        image=IMAGE_URI,
        command=["python", "-c"],
        args=[
            "import importlib.util, sys, pathlib; "
            "mods=['google.cloud.bigquery','google.cloud.bigquery_storage','pandas','pyarrow','db_dtypes','numpy','networkx','matplotlib','seaborn']; "
            "print('PY', sys.version); "
            "print('CHK', {m:('OK' if importlib.util.find_spec(m) else 'MISSING') for m in mods}); "
            "print('FILES', sorted(str(p) for p in pathlib.Path('.').glob('**/*.py'))[:50])"
        ],
    )

@dsl.container_component
def classify_comp(project_id: str, artifact_base_uri: str):
    return dsl.ContainerSpec(
        image=IMAGE_URI,
        command=["python", "components/classify.py"], 
        args=[
            "--project_id", project_id,
            "--artifact_base_uri", artifact_base_uri,
            "--input_dataset", "aml_prod",
            "--input_table", "txn_features_clean",
            "--output_dataset", "aml_prod",
            "--output_table", "df_txn_pred",
        ],
    )

@dsl.container_component
def subnet_comp(
    project_id: str,
    edgeset_dataset: str = "txn",
    edgeset_table: str = "txn_edgelist",
    pred_dataset: str = "aml_prod",
    pred_table: str = "df_txn_pred",
    output_dataset: str = "aml_prod",
    edges_output_table: str = "df_network_edges",
    nodes_output_table: str = "df_nw_txn_final",
    min_shared_nodes: int = 1,
    progress_every: int = 100,
):
    return dsl.ContainerSpec(
        image=IMAGE_URI,
        command=["python", "components/subnetwork.py"],
        args=[
            "--project_id", project_id,
            "--edgeset_dataset", edgeset_dataset,
            "--edgeset_table", edgeset_table,
            "--pred_dataset", pred_dataset,
            "--pred_table", pred_table,
            "--output_dataset", output_dataset,
            "--edges_output_table", edges_output_table,
            "--nodes_output_table", nodes_output_table,
            "--min_shared_nodes", str(min_shared_nodes),
            "--progress_every", str(progress_every),
        ],
    )

@dsl.container_component
def rank_comp(project_id: str):
    return dsl.ContainerSpec(
        image=IMAGE_URI,
        command=["python", "components/rank.py"], 
        args=[
            "--project_id", project_id,
            "--feats_dataset", "aml_prod",
            "--feats_table", "txn_features_clean",
            "--edges_dataset", "aml_prod",
            "--edges_table", "df_network_edges",
            "--nodes_dataset", "aml_prod",
            "--nodes_table", "df_nw_txn_final",
            "--output_dataset", "aml_prod",
            "--composite_output_table", "df_rank_composite",
            "--write_to_bq",
        ],
    )

@dsl.container_component
def summary_comp(project_id: str):
    return dsl.ContainerSpec(
        image=IMAGE_URI,
        command=["python", "components/summary.py"],\
        args=[
            "--project_id", project_id,
            "--rank_dataset", "aml_prod",
            "--rank_table", "df_rank_composite",
            "--actor_dataset", "actor",
            "--addrtxn_table", "addrtxn_edgelist",
            "--txaddr_table", "txaddr_edgelist",
            "--output_dataset", "aml_prod",
            "--summary_table", "df_subnetwork_summary",
            "--write_to_bq",
        ],
    )

@dsl.pipeline(name="aml-end-to-end", pipeline_root=PIPELINE_ROOT)
def aml_pipeline(project_id: str = PROJECT_ID, artifact_base_uri: str = PIPELINE_ROOT):
    d = diag_comp()
    s1 = classify_comp(project_id=project_id, artifact_base_uri=artifact_base_uri).after(d)
    s2 = subnet_comp(project_id=project_id).after(s1)  
    s3 = rank_comp(project_id=project_id).after(s2)
    s4 = summary_comp(project_id=project_id).after(s3)

if __name__ == "__main__":
    PACKAGE_PATH = "aml_pipeline.json"
    compiler.Compiler().compile(pipeline_func=aml_pipeline, package_path=PACKAGE_PATH)
    aiplatform.init(project=PROJECT_ID, location=REGION, staging_bucket="gs://thesis_model")
    job = aiplatform.PipelineJob(
        display_name="aml-end-to-end",
        template_path=PACKAGE_PATH,
        pipeline_root=PIPELINE_ROOT,
        enable_caching=False,
        parameter_values={},
    )
    job.run(service_account=f"vertex-pipelines@{PROJECT_ID}.iam.gserviceaccount.com")
