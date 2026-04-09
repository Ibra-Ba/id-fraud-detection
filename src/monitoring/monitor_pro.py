"""
Monitoring Evidently AI 0.7.x — IDNet Fraud Detector

Détecte :
  - Data drift      (statistiques pixels RGB)
  - Prediction drift (distribution genuine/fraud)
  - Qualité des données (nulls, types)
  - Métriques de performance (AUROC, F1)

Usage: python -m src.monitoring.monitor_pro \
  --use-mlflow-ref \
  --s3-prefix predictions/test_run
"""

import argparse
import json
import logging
import os
import tempfile
from pathlib import Path

import boto3
import mlflow
import pandas as pd
from dotenv import load_dotenv
from evidently import ColumnMapping
from evidently.metric_preset import (
    ClassificationPreset,
    DataDriftPreset,
    DataQualityPreset,
)
from evidently.report import Report
from mlflow.tracking import MlflowClient
from sklearn.metrics import f1_score, roc_auc_score

# ─────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DRIFT_SHARE_THRESHOLD = 0.3
REPORT_PATH = Path("reports/monitoring_report.html")

S3_BUCKET = os.getenv("S3_BUCKET")


# ─────────────────────────────────────────────────────────────
# 1. Load data
# ─────────────────────────────────────────────────────────────
def load_data(csv_path: Path) -> pd.DataFrame:

    df = pd.read_csv(csv_path)

    # 🔥 FIX TYPES
    df["score"] = df["score"].astype(float)
    df["prediction"] = df["prediction"].astype(int)

    if "label" in df.columns:
        df["label"] = df["label"].astype(int)

    logger.info(df.dtypes)

    if not {"prediction", "score"}.issubset(df.columns):
        raise ValueError(f"{csv_path} must contain prediction + score")

    return df


def load_reference_from_mlflow(model_name: str, alias: str = "champion") -> pd.DataFrame:
    logger.info(f"Loading reference dataset from MLflow ({model_name}@{alias})")

    client = MlflowClient()

    mv = client.get_model_version_by_alias(model_name, alias)
    run_id = mv.run_id

    logger.info(f"Downloading artifacts from run_id={run_id}")

    tmp_dir = tempfile.mkdtemp()

    client.download_artifacts(
        run_id, "reference_data/train_with_preds.csv", tmp_dir  # type: ignore
    )

    path = os.path.join(tmp_dir, "reference_data/train_with_preds.csv")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Reference dataset not found in MLflow at {path}")

    df = pd.read_csv(path)

    logger.info(f"Reference dataset loaded ({len(df)} rows)")
    return df


def load_s3_predictions(prefix: str) -> pd.DataFrame:
    logger.info(f"Loading S3 data from prefix: {prefix}")

    s3 = boto3.client("s3")
    records = []

    paginator = s3.get_paginator("list_objects_v2")

    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]

            if not key.endswith(".json"):
                continue

            data = s3.get_object(Bucket=S3_BUCKET, Key=key)
            content = json.loads(data["Body"].read())

            records.append(content)

    if records:

        df = pd.DataFrame(records)
        df = df.apply(pd.to_numeric, errors="ignore")
        logger.info(f"Loaded {len(records)} records from S3")
        return df
    else:
        raise ValueError(f"No data found in S3 prefix: {prefix}")


# ─────────────────────────────────────────────────────────────
# 2. Metrics
# ─────────────────────────────────────────────────────────────
def compute_metrics(df: pd.DataFrame) -> dict:
    if "label" not in df.columns:
        logger.warning("No labels → skipping performance metrics")
        return {}

    y_true = df["label"].values
    y_score = df["score"].values
    y_pred = df["prediction"].values

    return {
        "auroc": float(roc_auc_score(y_true, y_score)),  # type: ignore
        "f1": float(f1_score(y_true, y_pred)),  # type: ignore
        "fraud_rate": float(y_true.mean()),  # type: ignore
        "prediction_rate": float(y_pred.mean()),  # type: ignore
        "n_samples": int(len(df)),
    }


# ─────────────────────────────────────────────────────────────
# 3. Report
# ─────────────────────────────────────────────────────────────
def build_report(ref: pd.DataFrame, cur: pd.DataFrame) -> Report:

    # Supprimer colonne timestamp du rapport:
    for df in [ref, cur]:
        df.drop(columns=["timestamp"], errors="ignore", inplace=True)

    # Colonnes à exclure  du calcul numérique
    excluded_cols = ["label", "prediction", "id", "ID"]

    numerical_features = [
        col
        for col in ref.columns
        if pd.api.types.is_numeric_dtype(ref[col]) and col not in excluded_cols
    ]

    # Forcer conversion en numeric
    for col in numerical_features:
        ref[col] = pd.to_numeric(ref[col], errors="coerce")
        cur[col] = pd.to_numeric(cur[col], errors="coerce")
    # Supprimer colonnes constantes

    non_constant_cols = [col for col in numerical_features if ref[col].nunique() > 1]
    numerical_features = non_constant_cols

    column_mapping = ColumnMapping(
        target="label",
        prediction="prediction",
        numerical_features=numerical_features,
    )

    report = Report(
        metrics=[
            DataQualityPreset(),
            DataDriftPreset(),
            ClassificationPreset(),
        ]
    )

    # Supprimer les lignes avec des NaN
    ref = ref.dropna(subset=numerical_features)
    cur = cur.dropna(subset=numerical_features)

    report.run(
        reference_data=ref,
        current_data=cur,
        column_mapping=column_mapping,
    )
    logger.info(f"Numerical features used: {numerical_features}")
    return report


# 4. Drift extraction


def extract_drift(report: Report) -> dict:
    drift_detected = False
    drift_share_observed = 0.0

    try:
        report_dict = report.as_dict()
        for metric in report_dict.get("metrics", []):
            if metric.get("metric") == "DatasetDriftMetric":
                result = metric.get("result", {})

                drift_detected = result.get("dataset_drift", False)
                n_drifted = result.get("number_of_drifted_columns", 0)
                n_total = result.get("number_of_columns", 0)

                if n_total > 0:
                    drift_share_observed = n_drifted / n_total

                logger.info(f"Drift Stats: {n_drifted}/{n_total} columns drifted.")
                break
    except Exception as e:
        logger.warning(f"Drift extraction failed: {e}")

    # On déclenche l'alerte si le booléen est True
    # OU si le share observé dépasse le seuil 0.3
    drift_alert = drift_detected or (drift_share_observed >= DRIFT_SHARE_THRESHOLD)

    return {
        "drift_detected": drift_detected,
        "drift_share": float(drift_share_observed),
        "drift_alert": drift_alert,
    }


# 5. Pipeline


def run_monitoring(
    ref_path: Path,
    cur_path: Path | None = None,
    s3_prefix: str | None = None,
    log_mlflow: bool = True,
):
    logger.info("── Monitoring started ─────────────────────")
    if ref_path:
        ref = load_data(ref_path)
    else:
        logger.info("Loading reference from MLflow")
        ref = load_reference_from_mlflow(os.getenv("MLFLOW_MODEL_NAME", "IDNet-Fraud-Detector"))

    if s3_prefix:
        logger.info("Mode: S3")
        cur = load_s3_predictions(s3_prefix)
    elif cur_path:
        logger.info("Mode: LOCAL CSV")
        cur = load_data(cur_path)
    else:
        raise ValueError("Provide either --current or --s3-prefix")

    logger.info(f"Ref: {len(ref)} | Cur: {len(cur)}")

    metrics = compute_metrics(cur)

    report = build_report(ref, cur)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report.save_html(str(REPORT_PATH))

    drift = extract_drift(report)

    logger.info(f"Drift global={drift['drift_detected']} | " f"share={drift['drift_share']:.2%}")

    # MLflow

    if log_mlflow:
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))  # type: ignore

    #  HARD RESET CONTEXT
    if mlflow.active_run() is not None:
        logger.warning("Active MLflow run detected → closing it")
        mlflow.end_run()

    if "MLFLOW_RUN_ID" in os.environ:
        logger.warning("MLFLOW_RUN_ID found in env → removing it")
        os.environ.pop("MLFLOW_RUN_ID")

    mlflow.set_experiment("fraud-detection-monitoring")

    with mlflow.start_run(run_name="monitoring"):
        mlflow.log_metrics(
            {
                **metrics,
                "drift_share": drift["drift_share"],
            }
        )
        mlflow.log_artifact(str(REPORT_PATH))

    return {**metrics, **drift}


# CLI

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--reference", type=str, default=None)
    parser.add_argument("--use-mlflow-ref", action="store_true")
    parser.add_argument("--current", type=str, default=None)
    parser.add_argument("--s3-prefix", type=str, default=None)
    parser.add_argument("--no-mlflow", action="store_true")

    args = parser.parse_args()
    if not args.use_mlflow_ref and not args.reference:
        raise ValueError("Provide either --reference or --use-mlflow-ref")

    if not args.current and not args.s3_prefix:
        raise ValueError("Provide either --current or --s3-prefix")

    ref_path = None if args.use_mlflow_ref else Path(args.reference)

    results = run_monitoring(
        ref_path=ref_path,  # type: ignore
        cur_path=Path(args.current) if args.current else None,
        s3_prefix=args.s3_prefix,
        log_mlflow=not args.no_mlflow,
    )

    print("\n── Résumé ───────────────────────────────────")
    for k, v in results.items():
        print(f"{k}: {v}")
