"""
Pipeline de monitoring complet — IDNet Fraud Detector

Orchestre dans l'ordre :
  1. generate_predictions  (train_ref + test_run)
  2. build_reference       (reconstruit train_with_preds.csv depuis S3)
  3. log_reference         (log dans MLflow)
  4. simulate_threshold    (recalcule le seuil optimal)
  5. monitor_pro           (rapport Evidently + métriques MLflow)

Usage:
    python -m src.monitoring.run_monitoring_pipeline
    python -m src.monitoring.run_monitoring_pipeline --target-recall 0.95
    python -m src.monitoring.run_monitoring_pipeline --skip-generate
"""

import argparse
import logging
import os
from pathlib import Path

import mlflow
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient

from src.models.simulate_threshold import main as simulate

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROCESSED_DIR = Path(os.getenv("DATA_PROCESSED_DIR", "data/processed"))
MAX_SAMPLES = 500


def step_generate_predictions(target_recall: float):
    """Étape 1 — Génère les prédictions train_ref et test_run sur S3."""
    from src.monitoring.generate_predictions import run

    logger.info("── Étape 1/5 : Génération des prédictions ─────────────────")

    logger.info("Génération train_ref...")
    run(
        csv_path=str(PROCESSED_DIR / "train.csv"),
        prefix="predictions/train_ref",
        max_samples=MAX_SAMPLES,
    )

    logger.info("Génération test_run...")
    run(
        csv_path=str(PROCESSED_DIR / "test.csv"),
        prefix="predictions/test_run",
        max_samples=MAX_SAMPLES,
    )


def step_build_reference():
    """Étape 2 — Reconstruit train_with_preds.csv depuis S3."""
    from src.monitoring.build_reference_csv import main as build_reference

    logger.info("── Étape 2/5 : Build reference CSV ────────────────────────")
    build_reference()


def step_log_reference():
    """Étape 3 — Log le dataset de référence dans MLflow."""

    logger.info("── Étape 3/5 : Log référence dans MLflow ───────────────────")

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))  # type: ignore
    client = MlflowClient()
    model_name = os.getenv("MLFLOW_MODEL_NAME", "IDNet-Fraud-Detector")

    mv = client.get_model_version_by_alias(model_name, "champion")
    run_id = mv.run_id

    with mlflow.start_run(run_id=run_id):
        mlflow.log_artifact(
            str(PROCESSED_DIR / "train_with_preds.csv"),
            artifact_path="reference_data",
        )
        mlflow.log_param("reference_dataset", "train_with_preds_latest")

    logger.info(f"✅ Référence loggée sur run {run_id}")


def step_simulate_threshold(target_recall: float) -> float:
    """Étape 4 — Recalcule le seuil optimal et met à jour MLflow."""

    logger.info("── Étape 4/5 : Simulation seuil optimal ────────────────────")

    df = simulate(target_recall=target_recall)
    rec = df[df["fraud_recall"] >= target_recall]

    if rec.empty:
        logger.warning(f"Recall {target_recall:.0%} inatteignable — seuil conservé")
        return float(df.iloc[-1]["threshold"])

    best = rec.loc[rec["fraud_precision"].idxmax()]
    new_threshold = float(best["threshold"])

    logger.info(f"Nouveau seuil : {new_threshold} (recall={best['fraud_recall']:.2%})")

    # Met à jour le tag MLflow
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))  # type: ignore
    client = MlflowClient()
    model_name = os.getenv("MLFLOW_MODEL_NAME", "IDNet-Fraud-Detector")
    mv = client.get_model_version_by_alias(model_name, "champion")
    client.set_model_version_tag(model_name, mv.version, "optimal_threshold", str(new_threshold))
    logger.info(f"✅ Tag optimal_threshold={new_threshold} mis à jour sur v{mv.version}")

    return new_threshold


def step_monitor(log_mlflow: bool = True) -> dict:
    """Étape 5 — Rapport Evidently + métriques MLflow."""
    from src.monitoring.monitor_pro import run_monitoring

    logger.info("── Étape 5/5 : Monitoring Evidently ────────────────────────")

    return run_monitoring(
        ref_path=None,  # type: ignore
        s3_prefix="predictions/test_run",
        log_mlflow=log_mlflow,
    )


def run_pipeline(
    target_recall: float = 0.95,
    skip_generate: bool = False,
    log_mlflow: bool = True,
):
    logger.info("═══ Pipeline Monitoring démarré ════════════════════════════")

    if not skip_generate:
        step_generate_predictions(target_recall)
        step_build_reference()
        step_log_reference()

    step_simulate_threshold(target_recall)
    results = step_monitor(log_mlflow=log_mlflow)

    logger.info("═══ Pipeline Monitoring terminé ════════════════════════════")

    print("\n── Résumé final ─────────────────────────────────────────────────")
    for k, v in results.items():
        print(f"  {k}: {v}")

    if results.get("drift_alert"):
        logger.warning("⚠️  ALERTE DRIFT détectée\n" "   → Déclencher le CT via workflow_dispatch")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline monitoring IDNet")
    parser.add_argument(
        "--target-recall",
        type=float,
        default=0.95,
        help="Recall fraud minimum (défaut: 0.95)",
    )
    parser.add_argument(
        "--skip-generate",
        action="store_true",
        help="Saute la génération des prédictions (utilise les données S3 existantes)",
    )
    parser.add_argument(
        "--no-mlflow",
        dest="no_mlflow",
        action="store_true",
        help="Ne pas logger dans MLflow",
    )
    args = parser.parse_args()

    run_pipeline(
        target_recall=args.target_recall,
        skip_generate=args.skip_generate,
        log_mlflow=not args.no_mlflow,
    )
