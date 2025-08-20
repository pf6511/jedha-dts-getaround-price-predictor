import mlflow
import boto3
import os
from urllib.parse import urlparse

# ====== CONFIG ======
EXPERIMENT_NAME = "getaround_rental_price_predictor"
MLFLOW_TRACKING_URI = os.environ["MLFLOW_TRACKING_URI"]
ARTIFACT_ROOT = os.environ["ARTIFACT_ROOT"]
AWS_ACCESS_KEY_ID = os.environ["AWS_ACCESS_KEY_ID"]
AWS_SECRET_ACCESS_KEY = os.environ["AWS_SECRET_ACCESS_KEY"]

DRY_RUN = os.getenv("DRY_RUN", "true").lower() == "true"

# ====== INIT ======
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
client = mlflow.tracking.MlflowClient()

# Parse S3 info
parsed = urlparse(ARTIFACT_ROOT)
S3_BUCKET = parsed.netloc
S3_PREFIX_ROOT = parsed.path.lstrip("/").rstrip("/")

s3 = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY
)

# ====== Find experiment ======
exp = client.get_experiment_by_name(EXPERIMENT_NAME)
if not exp:
    raise ValueError(f"Experiment '{EXPERIMENT_NAME}' not found")
experiment_id = exp.experiment_id

print(f"🔍 Target experiment: '{EXPERIMENT_NAME}' (ID: {experiment_id})")
if DRY_RUN:
    print("🚧 DRY RUN mode is ON — nothing will be actually deleted.")

# ====== Delete runs and their artifacts ======
runs = client.search_runs([experiment_id], "", max_results=5000)

for run in runs:
    run_id = run.info.run_id
    print(f"\n📌 Found run: {run_id}")

    s3_prefix = f"{S3_PREFIX_ROOT}/{experiment_id}/{run_id}/"
    print(f"   S3 artifacts prefix: s3://{S3_BUCKET}/{s3_prefix}")

    # List S3 objects for run artifacts
    to_delete = s3.list_objects_v2(Bucket=S3_BUCKET, Prefix=s3_prefix)
    artifact_count = len(to_delete.get("Contents", []))
    print(f"   Found {artifact_count} artifact files in S3.")

    if not DRY_RUN:
        client.delete_run(run_id)
        if artifact_count > 0:
            for obj in to_delete["Contents"]:
                s3.delete_object(Bucket=S3_BUCKET, Key=obj["Key"])
        print("   ✅ Deleted run + artifacts.")

# ====== Delete models folder in S3 ======
models_prefix = f"{S3_PREFIX_ROOT}/{experiment_id}/models/"
print(f"\n📦 Checking for model registry artifacts: s3://{S3_BUCKET}/{models_prefix}")

models_list = s3.list_objects_v2(Bucket=S3_BUCKET, Prefix=models_prefix)
models_count = len(models_list.get("Contents", []))
if models_count > 0:
    print(f"   Found {models_count} model files.")
else:
    print("   No model registry artifacts found.")

if not DRY_RUN and models_count > 0:
    for obj in models_list["Contents"]:
        s3.delete_object(Bucket=S3_BUCKET, Key=obj["Key"])
    print("   ✅ Deleted model registry artifacts.")

# ====== Delete experiment from backend ======
if not DRY_RUN:
    client.delete_experiment(experiment_id)
    print(f"\n🗑️ Experiment {experiment_id} deleted successfully.")
else:
    print("\n✅ DRY RUN completed. No deletions were performed.")
