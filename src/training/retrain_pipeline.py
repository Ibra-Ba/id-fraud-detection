import logging

from src.models.evaluate import evaluate_from_run

# from src.models.register_model import register_and_promote
from src.models.train import train

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run():
    import os

    import mlflow

    logger.info("═══ Retrain pipeline démarré ════════════════════════════")
    reason = os.getenv("REASON", "unknown")

    # 1. Train
    run_id = train()
    mlflow.set_tag("retrain_reason", reason)
    logger.info(f"Run ID: {run_id}")

    # 2. Evaluate
    metrics = evaluate_from_run(run_id)

    # 3. Quality gate
    if metrics["gate_passed"]:
        logger.info("✅ Model validé → promotion")
        # register_and_promote(run_id)
    else:
        logger.warning("❌ Model rejeté (quality gate)")

    logger.info("═══ Retrain pipeline terminé ════════════════════════════")


if __name__ == "__main__":
    run()
