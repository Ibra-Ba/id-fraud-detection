"""
Pipeline de monitoring complet — IDNet Fraud Detector

Orchestre dans l'ordre :
  1. generate_predictions  (train_ref + test_run)  — optionnel via --skip-generate
  2. build_reference       (reconstruit train_with_preds.csv depuis S3)
  3. simulate_threshold    (recalcule le seuil optimal)
  4. monitor_pro           (rapport Evidently + métriques MLflow)

Note : le log de la référence dans MLflow est géré par cd.yml
(job refresh-reference) à chaque nouveau champion déployé.
Ce pipeline ne modifie jamais le run champion Bac+4.

Usage:
    python -m src.monitoring.run_monitoring_pipeline
    python -m src.monitoring.run_monitoring_pipeline --target-recall 0.95
    python -m src.monitoring.run_monitoring_pipeline --skip-generate
"""

import argparse
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

from src.models.simulate_threshold import main as simulate

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROCESSED_DIR = Path(os.getenv("DATA_PROCESSED_DIR", "data/processed"))
MAX_SAMPLES = 500


# ─────────────────────────────────────────────────────────────
# Étapes du pipeline
# ─────────────────────────────────────────────────────────────


def step_generate_predictions(target_recall: float):
    """Étape 1 — Génère les prédictions train_ref et test_run sur S3."""
    from src.monitoring.generate_predictions import run

    logger.info("── Étape 1/4 : Génération des prédictions ──────────────────")

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

    logger.info("── Étape 2/4 : Build reference CSV ─────────────────────────")
    build_reference()


def step_simulate_threshold(target_recall: float):
    """Étape 3 — Recalcule le seuil optimal et le logue dans MLflow monitoring."""
    logger.info("── Étape 3/4 : Simulation seuil optimal ────────────────────")

    simulate(
        target_recall=target_recall,
        source="csv",
        csv_path=str(PROCESSED_DIR / "train_with_preds.csv"),
    )

    logger.info("Threshold recalculé et loggué dans MLflow monitoring")


def step_monitor(log_mlflow: bool = True) -> dict:
    """Étape 4 — Rapport Evidently + métriques MLflow."""
    from src.monitoring.monitor_pro import run_monitoring

    logger.info("── Étape 4/4 : Monitoring Evidently ────────────────────────")

    return run_monitoring(
        ref_path=None,
        s3_prefix="predictions/test_run",
        log_mlflow=log_mlflow,
    )


# ─────────────────────────────────────────────────────────────
# Pipeline complet
# ─────────────────────────────────────────────────────────────


def run_pipeline(
    target_recall: float = 0.95,
    skip_generate: bool = False,
    log_mlflow: bool = True,
) -> dict:
    logger.info("═══ Pipeline Monitoring démarré ════════════════════════════")

    if not skip_generate:
        step_generate_predictions(target_recall)
        step_build_reference()

    step_simulate_threshold(target_recall)
    results = step_monitor(log_mlflow=log_mlflow)

    logger.info("═══ Pipeline Monitoring terminé ════════════════════════════")

    print("\n── Résumé final ─────────────────────────────────────────────────")
    for k, v in results.items():
        print(f"  {k}: {v}")

    # Déclenchement automatique du CT si drift détecté
    if results.get("drift_alert"):
        logger.warning("Drift détecté")

        auto_retrain = os.getenv("AUTO_RETRAIN", "0") == "1"
        if auto_retrain:
            logger.info("Déclenchement du CT workflow...")
            _trigger_ct_workflow()
        else:
            logger.info("AUTO_RETRAIN désactivé (set AUTO_RETRAIN=1 pour activer)")

    return results


def _trigger_ct_workflow():
    """Déclenche ct.yml via l'API GitHub."""
    try:
        import requests

        repo = os.getenv("GITHUB_REPOSITORY")
        token = os.getenv("GITHUB_TOKEN")

        if not repo or not token:
            raise ValueError("GITHUB_REPOSITORY ou GITHUB_TOKEN manquant")

        url = f"https://api.github.com/repos/{repo}/actions/workflows/ct.yml/dispatches"
        response = requests.post(
            url,
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github+json",
            },
            json={"ref": "main", "inputs": {"reason": "data_drift_detected"}},
        )
        if response.status_code == 204:
            logger.info("CT workflow déclenché")
        else:
            logger.error(f"Échec déclenchement CT : {response.status_code} {response.text}")
    except Exception as e:
        logger.error(f"Erreur déclenchement CT : {e}")


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

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
