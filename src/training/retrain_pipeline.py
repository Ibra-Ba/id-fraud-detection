"""
Pipeline de réentraînement — IDNet Fraud Detector Bac+5 (stub)

Ce module implémente le Continuous Training du Bac+5.
Le Bac+5 ne réentraîne pas le modèle : il consomme le champion
produit par le Bac+4 et vérifie qu'il satisfait toujours les
critères métier (AUROC >= seuil, recall fraude >= 95%).

Logique CT :
  1. Charge le champion @champion depuis MLflow (Bac+4)
  2. Vérifie ses métriques : AUROC, threshold, recall
  3. Logue le résultat dans l'expérience Bac+5 (idnet-monitoring)
  4. Écrit les outputs GitHub Actions pour le job promote

Pourquoi un stub et non un vrai réentraînement :
  - Le modèle EfficientNet-B0 est entraîné dans le Bac+4 (bloc6-idnet)
  - Le Bac+5 est la couche MLOps qui consomme ce modèle
  - Le CT se déclenche sur drift détecté → vérifie si le champion
    actuel tient toujours, sinon signale qu'un réentraînement Bac+4
    est nécessaire (hors scope automatisé)

Outputs GitHub Actions (écrits dans GITHUB_OUTPUT) :
  new_run_id      : run_id du champion vérifié
  new_auroc       : AUROC du champion
  champion_auroc  : identique (pas de nouveau modèle)
  promoted        : toujours "false" (pas de nouvelle version)

Usage:
    python -m src.training.retrain_pipeline
"""

import logging
import os
import sys

import mlflow
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Constantes ────────────────────────────────────────────────────────────────
MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "IDNet-Fraud-Detector")
MONITORING_EXPERIMENT = "idnet-monitoring"

# Seuils de qualité minimaux (critères métier KYC)
MIN_AUROC = float(os.getenv("MIN_AUROC", "0.95"))
MIN_RECALL = float(os.getenv("MIN_RECALL", "0.95"))


# ── Vérification du champion ──────────────────────────────────────────────────


def check_champion_health(client: MlflowClient) -> dict:
    """
    Vérifie que le champion @champion satisfait les critères métier.

    Returns:
        dict avec run_id, version, auroc, threshold, recall, healthy
    """
    try:
        mv = client.get_model_version_by_alias(MODEL_NAME, "champion")
    except Exception as e:
        raise RuntimeError(f"Champion introuvable dans MLflow ({MODEL_NAME}@champion) : {e}") from e

    run = client.get_run(mv.run_id)
    metrics = run.data.metrics
    tags = run.data.tags

    # AUROC : test_auroc en priorité, vl_auroc en fallback
    auroc = metrics.get("test_auroc", metrics.get("vl_auroc", 0.0))

    # Threshold : metrics → tags (cohérent avec register_model.py Bac+4)
    threshold = metrics.get("optimal_threshold")
    if threshold is None:
        threshold = float(tags.get("optimal_threshold", "0.5"))

    # Recall : loggué par evaluate.py Bac+4 sous test_recall si disponible
    recall = metrics.get("test_recall", metrics.get("threshold_recall", None))

    healthy = auroc >= MIN_AUROC
    if recall is not None:
        healthy = healthy and (recall >= MIN_RECALL)

    result = {
        "run_id": mv.run_id,
        "version": mv.version,
        "auroc": float(auroc),
        "threshold": float(threshold),
        "recall": float(recall) if recall is not None else None,
        "healthy": healthy,
        "origin": tags.get("origin", "unknown"),
    }

    logger.info(
        f"Champion v{mv.version} | AUROC={auroc:.4f} | "
        f"threshold={threshold:.4f} | "
        f"recall={'N/A' if recall is None else f'{recall:.4f}'} | "
        f"healthy={healthy}"
    )

    return result


# ── Log dans MLflow ───────────────────────────────────────────────────────────


def log_ct_run(health: dict, reason: str) -> str:
    """
    Logue le résultat du CT dans l'expérience monitoring.
    Retourne le run_id du log.
    """
    # Clôture tout run actif (pattern monitor_pro.py)
    if mlflow.active_run():
        logger.warning("Run MLflow actif détecté → fermeture")
        mlflow.end_run()
    os.environ.pop("MLFLOW_RUN_ID", None)

    mlflow.set_experiment(MONITORING_EXPERIMENT)

    with mlflow.start_run(run_name="ct_check") as run:
        mlflow.set_tag("trigger_reason", reason)
        mlflow.set_tag("model_name", MODEL_NAME)
        mlflow.set_tag("champion_version", health["version"])
        mlflow.set_tag("champion_run_id", health["run_id"])
        mlflow.set_tag("champion_origin", health["origin"])
        mlflow.set_tag("ct_mode", "stub_bac5")
        mlflow.set_tag("champion_healthy", str(health["healthy"]))

        metrics_to_log = {
            "champion_auroc": health["auroc"],
            "champion_threshold": health["threshold"],
            "min_auroc_threshold": MIN_AUROC,
            "min_recall_threshold": MIN_RECALL,
        }
        if health["recall"] is not None:
            metrics_to_log["champion_recall"] = health["recall"]

        mlflow.log_metrics(metrics_to_log)

        ct_run_id = run.info.run_id

    logger.info(f"CT loggué dans MLflow (run_id={ct_run_id})")
    return ct_run_id


# ── Outputs GitHub Actions ────────────────────────────────────────────────────


def write_github_outputs(health: dict) -> None:
    """
    Écrit les outputs attendus par ct.yml dans GITHUB_OUTPUT.

    Le job promote dans ct.yml lit :
      new_run_id, new_auroc, champion_auroc, promoted
    """
    github_output = os.environ.get("GITHUB_OUTPUT")
    if not github_output:
        logger.warning("GITHUB_OUTPUT non défini — outputs non écrits (mode local ?)")
        return

    with open(github_output, "a") as f:
        # Le CT stub ne crée pas de nouveau modèle → promoted=false
        f.write(f"new_run_id={health['run_id']}\n")
        f.write(f"new_auroc={health['auroc']:.4f}\n")
        f.write(f"champion_auroc={health['auroc']:.4f}\n")
        f.write("promoted=false\n")

    logger.info("Outputs GitHub Actions écrits")


# ── Point d'entrée principal ──────────────────────────────────────────────────


def main() -> dict:
    """
    Pipeline CT complet :
      1. Vérifie le champion
      2. Logue dans MLflow
      3. Écrit les outputs GitHub Actions
      4. Exit non-zéro si le champion est dégradé (alerte CI)

    Returns:
        dict avec les résultats du check
    """
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    client = MlflowClient()

    reason = os.getenv("REASON", "manual trigger")
    logger.info(f"═══ CT Pipeline démarré (raison: {reason}) ═══════════════")

    # 1. Vérifie le champion
    try:
        health = check_champion_health(client)
    except RuntimeError as e:
        logger.error(str(e))
        sys.exit(1)

    # 2. Logue dans MLflow
    log_ct_run(health, reason)

    # 3. Outputs GitHub Actions
    write_github_outputs(health)

    # 4. Résumé
    logger.info("═══ CT Pipeline terminé ════════════════════════════════════")

    print("\n── Résumé CT ────────────────────────────────────────────────")
    print(f"  Champion       : {MODEL_NAME} v{health['version']}")
    print(f"  AUROC          : {health['auroc']:.4f} (min requis : {MIN_AUROC})")
    if health["recall"] is not None:
        print(f"  Recall         : {health['recall']:.4f} (min requis : {MIN_RECALL})")
    print(f"  Threshold      : {health['threshold']:.4f}")
    print(f"  Healthy        : {health['healthy']}")
    print("  Promoted       : false (stub — réentraînement via Bac+4)")

    if not health["healthy"]:
        logger.error(
            f"Champion dégradé (AUROC={health['auroc']:.4f} < {MIN_AUROC}). "
            f"Un réentraînement manuel via le repo Bac+4 (bloc6-idnet) est nécessaire."
        )
        sys.exit(2)

    return health


if __name__ == "__main__":
    main()
