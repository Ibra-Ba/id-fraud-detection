"""
Bootstrap de la référence monitoring — IDNet Fraud Detector Bac+5

À lancer une seule fois depuis le repo Bac+5 pour initialiser
la référence dans MLflow monitoring.

Charge train_with_preds.csv depuis S3 (cni-fraud-detection)
et le logue dans l'expérience idnet-monitoring.

Usage:
    python scripts/bootstrap_reference.py
"""

import os
import tempfile

import boto3
import mlflow
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient

load_dotenv()

S3_BUCKET = "cni-fraud-detection"
S3_KEY = "data/processed/train_with_preds.csv"
MODEL_NAME = "IDNet-Fraud-Detector"
MONITORING_EXPERIMENT = "idnet-monitoring"


def main():
    print("── Bootstrap référence monitoring ──────────────────────────────")

    # ── 1. Tracking URI ───────────────────────────────────────────────────
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        raise RuntimeError("MLFLOW_TRACKING_URI non défini dans .env")
    print(f"MLflow URI : {tracking_uri}")

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    # ── 2. Vérifie le champion ────────────────────────────────────────────
    print(f"\nChargement du champion {MODEL_NAME}@champion...")
    mv = client.get_model_version_by_alias(MODEL_NAME, "champion")
    run = client.get_run(mv.run_id)

    champion_version = mv.version
    auroc = run.data.metrics.get("vl_auroc", run.data.metrics.get("test_auroc", 0.0))
    threshold = run.data.metrics.get("optimal_threshold")
    if threshold is None:
        threshold = float(run.data.tags.get("optimal_threshold", "0.5"))

    print(f"  version   : {champion_version}")
    print(f"  run_id    : {mv.run_id}")
    print(f"  auroc     : {auroc:.4f}")
    print(f"  threshold : {threshold:.4f}")

    # ── 3. Télécharge train_with_preds.csv depuis S3 ──────────────────────
    print(f"\nTéléchargement depuis s3://{S3_BUCKET}/{S3_KEY}...")
    s3 = boto3.client("s3")
    tmp = tempfile.mktemp(suffix=".csv")
    s3.download_file(S3_BUCKET, S3_KEY, tmp)
    print(f"  Fichier téléchargé : {tmp}")

    # ── 4. Logue dans MLflow monitoring ───────────────────────────────────
    print(f"\nLog dans MLflow expérience '{MONITORING_EXPERIMENT}'...")

    if mlflow.active_run():
        mlflow.end_run()
    os.environ.pop("MLFLOW_RUN_ID", None)

    mlflow.set_experiment(MONITORING_EXPERIMENT)

    with mlflow.start_run(run_name="reference_update") as run:
        mlflow.set_tag("reference_version", str(champion_version))
        mlflow.set_tag("reference_type", "train_with_preds")
        mlflow.set_tag("model_name", MODEL_NAME)
        mlflow.set_tag("champion_auroc", str(auroc))
        mlflow.set_tag("source", "bootstrap")
        mlflow.log_metric("champion_version_int", int(champion_version))
        mlflow.log_metric("champion_auroc_metric", float(auroc))
        mlflow.log_artifact(tmp, artifact_path="reference_data")

        run_id = run.info.run_id

    print("\nReference bootstrapée avec succes")
    print(f"  experiment : {MONITORING_EXPERIMENT}")
    print(f"  run_id     : {run_id}")
    print(f"  reference_version : {champion_version}")
    print("\nLe pipeline monitoring peut maintenant tourner.")


if __name__ == "__main__":
    main()
