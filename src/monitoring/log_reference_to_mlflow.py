import os

import mlflow
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient

load_dotenv()
MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "IDNet-Fraud-Detector")
ALIAS = "champion"

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))  # type: ignore

client = MlflowClient()
mv = client.get_model_version_by_alias(MODEL_NAME, ALIAS)
run_id = mv.run_id

with mlflow.start_run(run_id=run_id):
    mlflow.log_artifact("data/processed/train_with_preds.csv", artifact_path="reference_data")
    mlflow.log_param("reference_dataset", "train_with_preds_v1")

print(f"Reference dataset loggé sur run {run_id}")
