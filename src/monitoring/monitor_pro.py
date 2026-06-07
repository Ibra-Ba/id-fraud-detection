"""
Monitoring Evidently AI 0.7.x — IDNet Fraud Detector

Détecte :
  - Data drift      (statistiques pixels RGB)
  - Prediction drift (distribution genuine/fraud)
  - Qualité des données (nulls, types)
  - Métriques de performance (AUROC, F1)

Chargement de la référence :
  1. MLflow monitoring — run le plus récent tagué reference_version
     (loggué par cd.yml job refresh-reference, jamais le run Bac+4)
  2. Fallback S3 — data/processed/train_with_preds.csv
     (si MLflow indisponible ou pas encore initialisé)

Usage: python -m src.monitoring.monitor_pro \
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

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DRIFT_SHARE_THRESHOLD = 0.3
REPORT_PATH = Path("reports/monitoring_report.html")
MONITORING_EXPERIMENT = "idnet-monitoring"
S3_BUCKET = os.getenv("S3_BUCKET")
S3_REFERENCE_KEY = "data/processed/train_with_preds.csv"


# ─────────────────────────────────────────────────────────────
# 1. Chargement de la référence
# ─────────────────────────────────────────────────────────────


def load_reference_from_mlflow_monitoring() -> tuple[pd.DataFrame, str]:
    """
    Charge la référence depuis l'expérience MLflow monitoring.
    Cherche le run le plus récent tagué reference_version.
    Ne touche jamais au run champion Bac+4.
    """
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    client = MlflowClient()

    experiment = client.get_experiment_by_name(MONITORING_EXPERIMENT)
    if experiment is None:
        raise ValueError(
            f"Expérience '{MONITORING_EXPERIMENT}' introuvable dans MLflow. "
            f"Lancer cd.yml une première fois pour initialiser la référence."
        )

    # Cherche les runs avec le tag reference_version, triés par date décroissante
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="tags.reference_type = 'train_with_preds'",
        order_by=["start_time DESC"],
        max_results=1,
    )

    if not runs:
        raise ValueError(
            "Aucun run de référence trouvé dans MLflow monitoring. "
            "Le job refresh-reference de cd.yml n'a pas encore tourné."
        )

    ref_run = runs[0]
    ref_version = ref_run.data.tags.get("reference_version", "unknown")
    run_id = ref_run.info.run_id

    logger.info(f"Référence trouvée : run_id={run_id} | " f"reference_version={ref_version}")

    tmp_dir = tempfile.mkdtemp()
    client.download_artifacts(run_id, "reference_data/train_with_preds.csv", tmp_dir)
    path = os.path.join(tmp_dir, "reference_data", "train_with_preds.csv")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Artifact train_with_preds.csv introuvable dans le run {run_id}")

    df = pd.read_csv(path)
    df = _clean_reference(df)
    logger.info(f"Référence MLflow chargée : {len(df)} lignes (v{ref_version})")
    return df, ref_version


def _clean_reference(df):
    """Supprime les colonnes du manifest Bac+4 inutiles pour Evidently."""
    return df.drop(columns=["path", "image_path"], errors="ignore")


def load_reference_from_s3() -> tuple[pd.DataFrame, str]:
    """
    Fallback : charge la référence directement depuis S3.
    Utilisé si MLflow est indisponible ou pas encore initialisé.
    """
    if not S3_BUCKET:
        raise RuntimeError("S3_BUCKET non défini")

    logger.info(f"Fallback S3 : chargement depuis s3://{S3_BUCKET}/{S3_REFERENCE_KEY}")
    s3 = boto3.client("s3")

    tmp = tempfile.mktemp(suffix=".csv")
    s3.download_file(S3_BUCKET, S3_REFERENCE_KEY, tmp)

    df = _clean_reference(pd.read_csv(tmp))
    logger.info(f"Référence S3 chargée : {len(df)} lignes")
    return df, "s3_fallback"


def load_reference(ref_path: Path | None = None) -> tuple[pd.DataFrame, str]:
    """
    Charge la référence avec priorité :
      1. ref_path local (mode dev / test)
      2. MLflow monitoring (production)
      3. S3 fallback

    Returns (DataFrame, version_label)
    """
    if ref_path is not None:
        logger.info(f"Référence locale : {ref_path}")
        df = _clean_reference(pd.read_csv(ref_path))
        return df, "local"

    try:
        return load_reference_from_mlflow_monitoring()
    except Exception as e:
        logger.warning(f"MLflow monitoring indisponible : {e} → fallback S3")

    return load_reference_from_s3()


# ─────────────────────────────────────────────────────────────
# 2. Chargement des données courantes
# ─────────────────────────────────────────────────────────────


def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["score"] = df["score"].astype(float)
    df["prediction"] = df["prediction"].astype(int)
    if "label" in df.columns:
        df["label"] = df["label"].astype(int)
    if not {"prediction", "score"}.issubset(df.columns):
        raise ValueError(f"{csv_path} doit contenir prediction + score")
    return df


def load_s3_predictions(prefix: str) -> pd.DataFrame:
    logger.info(f"Chargement S3 : {prefix}")
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

    if not records:
        raise ValueError(f"Aucune donnée dans S3 prefix : {prefix}")

    df = pd.DataFrame(records)
    df = df.apply(pd.to_numeric, errors="ignore")
    logger.info(f"{len(records)} records chargés depuis S3")
    return df


# ─────────────────────────────────────────────────────────────
# 3. Métriques
# ─────────────────────────────────────────────────────────────


def compute_metrics(df: pd.DataFrame) -> dict:
    if "label" not in df.columns:
        logger.warning("Pas de labels → métriques de performance ignorées")
        return {}

    y_true = df["label"].values
    y_score = df["score"].values
    y_pred = df["prediction"].values

    return {
        "auroc": float(roc_auc_score(y_true, y_score)),
        "f1": float(f1_score(y_true, y_pred)),
        "fraud_rate": float(y_true.mean()),
        "prediction_rate": float(y_pred.mean()),
        "n_samples": int(len(df)),
    }


# ─────────────────────────────────────────────────────────────
# 4. Rapport Evidently
# ─────────────────────────────────────────────────────────────


def build_report(ref: pd.DataFrame, cur: pd.DataFrame) -> Report:
    for df in [ref, cur]:
        df.drop(columns=["timestamp"], errors="ignore", inplace=True)

    excluded_cols = ["label", "prediction", "id", "ID"]
    numerical_features = [
        col
        for col in ref.columns
        if pd.api.types.is_numeric_dtype(ref[col]) and col not in excluded_cols
    ]

    for col in numerical_features:
        ref[col] = pd.to_numeric(ref[col], errors="coerce")
        cur[col] = pd.to_numeric(cur[col], errors="coerce")

    numerical_features = [col for col in numerical_features if ref[col].nunique() > 1]

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

    ref = ref.dropna(subset=numerical_features)
    cur = cur.dropna(subset=numerical_features)

    report.run(
        reference_data=ref,
        current_data=cur,
        column_mapping=column_mapping,
    )
    logger.info(f"Features numériques utilisées : {numerical_features}")
    return report


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
                logger.info(f"Drift : {n_drifted}/{n_total} colonnes driftées")
                break
    except Exception as e:
        logger.warning(f"Extraction drift échouée : {e}")

    drift_alert = drift_detected or (drift_share_observed >= DRIFT_SHARE_THRESHOLD)
    return {
        "drift_detected": drift_detected,
        "drift_share": float(drift_share_observed),
        "drift_alert": drift_alert,
    }


# ─────────────────────────────────────────────────────────────
# 5. Pipeline principal
# ─────────────────────────────────────────────────────────────


def run_monitoring(
    ref_path: Path | None = None,
    cur_path: Path | None = None,
    s3_prefix: str | None = None,
    log_mlflow: bool = True,
) -> dict:
    logger.info("── Monitoring démarré ─────────────────────────────────────")

    # Référence : MLflow monitoring → fallback S3 → local
    ref, ref_version = load_reference(ref_path)

    # Données courantes
    if s3_prefix:
        cur = load_s3_predictions(s3_prefix)
    elif cur_path:
        cur = load_data(cur_path)
    else:
        raise ValueError("Fournir --current ou --s3-prefix")

    logger.info(f"Ref : {len(ref)} lignes (v{ref_version}) | Cur : {len(cur)} lignes")

    metrics = compute_metrics(cur)
    report = build_report(ref, cur)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report.save_html(str(REPORT_PATH))

    drift = extract_drift(report)
    logger.info(f"Drift global={drift['drift_detected']} | share={drift['drift_share']:.2%}")

    if log_mlflow:
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

        if mlflow.active_run():
            logger.warning("Run MLflow actif détecté → fermeture")
            mlflow.end_run()
        os.environ.pop("MLFLOW_RUN_ID", None)

        mlflow.set_experiment(MONITORING_EXPERIMENT)

        with mlflow.start_run(run_name="monitoring"):
            mlflow.set_tag("model_name", os.getenv("MLFLOW_MODEL_NAME"))
            mlflow.set_tag("reference_version", ref_version)
            mlflow.log_metrics(
                {
                    **metrics,
                    "drift_share": drift["drift_share"],
                }
            )
            mlflow.log_artifact(str(REPORT_PATH))

    return {**metrics, **drift}


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reference",
        type=str,
        default=None,
        help="CSV local (dev uniquement — prod charge depuis MLflow/S3)",
    )
    parser.add_argument("--current", type=str, default=None)
    parser.add_argument("--s3-prefix", type=str, default=None)
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args()

    if not args.current and not args.s3_prefix:
        raise ValueError("Fournir --current ou --s3-prefix")

    results = run_monitoring(
        ref_path=Path(args.reference) if args.reference else None,
        cur_path=Path(args.current) if args.current else None,
        s3_prefix=args.s3_prefix,
        log_mlflow=not args.no_mlflow,
    )

    print("\n── Résumé ───────────────────────────────────")
    for k, v in results.items():
        print(f"  {k}: {v}")
