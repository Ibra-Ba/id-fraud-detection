"""
Simulation du seuil optimal — IDNet Fraud Detector Bac+5

Recalcule le seuil optimal (recall >= target) depuis les prédictions
du champion MLflow et le logue dans l'expérience de monitoring.

Cohérence Bac+4 :
  - Même logique que find_optimal_threshold() dans train.py :
    precision_recall_curve + contrainte recall >= target_recall
  - Lit optimal_threshold depuis metrics en priorité, tags en fallback
  - Écrit le résultat dans MLflow (expérience monitoring, pas le run champion)

Usage:
    python -m src.models.simulate_threshold
    python -m src.models.simulate_threshold --target-recall 0.95
    python -m src.models.simulate_threshold --source mlflow
    python -m src.models.simulate_threshold --source csv --csv-path data/processed/val.csv
"""

import argparse
import logging
import os
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient
from sklearn.metrics import precision_recall_curve

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Constantes ────────────────────────────────────────────────────────────────
MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "IDNet-Fraud-Detector")
MONITORING_EXPERIMENT = "idnet-monitoring"
TARGET_RECALL_DEFAULT = 0.95


# ── Calcul du seuil ───────────────────────────────────────────────────────────


def find_optimal_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    target_recall: float = TARGET_RECALL_DEFAULT,
) -> dict:
    """
    Seuil maximisant la précision sous contrainte recall >= target_recall.
    Logique identique à train.py Bac+4.

    Retourne un dict avec threshold, precision, recall atteints.
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_score)

    # precision_recall_curve retourne n+1 points, thresholds a n valeurs
    valid = [
        (p, r, t)
        for p, r, t in zip(precisions[:-1], recalls[:-1], thresholds, strict=False)
        if r >= target_recall
    ]

    if not valid:
        logger.warning(
            f"Recall cible {target_recall:.0%} inatteignable — "
            f"recall max disponible : {recalls.max():.4f}. "
            f"Seuil minimal retourné."
        )
        return {
            "threshold": float(thresholds[0]),
            "precision": float(precisions[0]),
            "recall": float(recalls[0]),
            "target_recall": target_recall,
            "target_reached": False,
        }

    best_precision, best_recall, best_threshold = max(valid, key=lambda x: x[0])
    logger.info(
        f"Seuil optimal : {best_threshold:.4f} "
        f"(precision={best_precision:.4f}, recall={best_recall:.4f})"
    )
    return {
        "threshold": float(best_threshold),
        "precision": float(best_precision),
        "recall": float(best_recall),
        "target_recall": target_recall,
        "target_reached": True,
    }


# ── Sources de données ────────────────────────────────────────────────────────


def load_from_mlflow_artifacts(client: MlflowClient, run_id: str) -> pd.DataFrame:
    """Charge train_with_preds.csv depuis les artifacts MLflow du champion."""
    import tempfile

    tmp_dir = tempfile.mkdtemp()
    try:
        client.download_artifacts(run_id, "reference_data/train_with_preds.csv", tmp_dir)
        path = os.path.join(tmp_dir, "reference_data", "train_with_preds.csv")
        df = pd.read_csv(path)
        logger.info(f"Référence chargée depuis MLflow ({len(df)} lignes)")
        return df
    except Exception as e:
        raise RuntimeError(f"Impossible de charger les artifacts MLflow : {e}") from e


def load_from_csv(csv_path: str) -> pd.DataFrame:
    """Charge un CSV local avec colonnes label + score."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV introuvable : {path}")
    df = pd.read_csv(path)
    logger.info(f"CSV chargé : {path} ({len(df)} lignes)")
    return df


def validate_dataframe(df: pd.DataFrame) -> None:
    """Vérifie que le DataFrame contient les colonnes requises."""
    required = {"label", "score"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Colonnes manquantes dans le CSV : {missing}. "
            f"Colonnes disponibles : {list(df.columns)}"
        )
    if df["label"].isna().any() or df["score"].isna().any():
        raise ValueError("Des valeurs NaN trouvées dans label ou score.")


# ── Pipeline principal ────────────────────────────────────────────────────────


def main(
    target_recall: float = TARGET_RECALL_DEFAULT,
    source: str = "mlflow",
    csv_path: str | None = None,
) -> dict:
    """
    Recalcule le seuil optimal et le logue dans MLflow monitoring.

    Args:
        target_recall: recall fraude minimum à garantir (défaut 0.95)
        source: "mlflow" (artifacts du champion) ou "csv" (fichier local)
        csv_path: chemin CSV si source="csv"

    Returns:
        dict avec threshold, precision, recall, target_reached
    """
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    client = MlflowClient()

    # ── 1. Charge le champion ─────────────────────────────────────────────────
    logger.info(f"Chargement du champion : {MODEL_NAME}@champion")
    try:
        mv = client.get_model_version_by_alias(MODEL_NAME, "champion")
    except Exception as e:
        raise RuntimeError(f"Champion introuvable dans MLflow : {e}") from e

    champion_run_id = mv.run_id
    champion_version = mv.version
    champion_run = client.get_run(champion_run_id)

    # Threshold actuel du champion (pour comparaison dans le log)
    current_threshold = champion_run.data.metrics.get("optimal_threshold")
    if current_threshold is None:
        current_threshold = float(champion_run.data.tags.get("optimal_threshold", "0.5"))

    logger.info(
        f"Champion : v{champion_version} | run_id={champion_run_id} | "
        f"threshold actuel={current_threshold:.4f}"
    )

    # ── 2. Charge les données ─────────────────────────────────────────────────
    if source == "mlflow":
        df = load_from_mlflow_artifacts(client, champion_run_id)
    elif source == "csv":
        if not csv_path:
            raise ValueError("--csv-path requis quand --source=csv")
        df = load_from_csv(csv_path)
    else:
        raise ValueError(f"Source inconnue : {source}. Valeurs valides : mlflow, csv")

    validate_dataframe(df)

    y_true = df["label"].astype(int).values
    y_score = df["score"].astype(float).values

    # ── 3. Calcule le seuil ───────────────────────────────────────────────────
    result = find_optimal_threshold(y_true, y_score, target_recall)

    logger.info(
        f"Threshold recalculé : {result['threshold']:.4f} "
        f"(était : {current_threshold:.4f}, "
        f"delta : {result['threshold'] - current_threshold:+.4f})"
    )

    # ── 4. Logue dans MLflow (expérience monitoring, pas le run champion) ─────
    # Clôture tout run actif pour éviter les conflits (pattern monitor_pro.py)
    if mlflow.active_run():
        logger.warning("Run MLflow actif détecté → fermeture")
        mlflow.end_run()
    os.environ.pop("MLFLOW_RUN_ID", None)

    mlflow.set_experiment(MONITORING_EXPERIMENT)

    with mlflow.start_run(run_name="simulate_threshold"):
        mlflow.set_tag("model_name", MODEL_NAME)
        mlflow.set_tag("model_version", champion_version)
        mlflow.set_tag("champion_run_id", champion_run_id)
        mlflow.set_tag("source", source)
        mlflow.set_tag("target_reached", str(result["target_reached"]))

        mlflow.log_metrics(
            {
                "optimal_threshold": result["threshold"],
                "threshold_precision": result["precision"],
                "threshold_recall": result["recall"],
                "target_recall": target_recall,
                "previous_threshold": current_threshold,
                "threshold_delta": result["threshold"] - current_threshold,
            }
        )

    logger.info("Threshold loggué dans MLflow (expérience monitoring)")

    if not result["target_reached"]:
        logger.warning(
            f"ATTENTION : recall cible {target_recall:.0%} non atteint. "
            f"Recall obtenu : {result['recall']:.4f}. "
            f"Vérifier la distribution des données."
        )

    return result


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Recalcule le seuil optimal du champion IDNet")
    parser.add_argument(
        "--target-recall",
        type=float,
        default=TARGET_RECALL_DEFAULT,
        help=f"Recall fraude minimum (défaut: {TARGET_RECALL_DEFAULT})",
    )
    parser.add_argument(
        "--source",
        choices=["mlflow", "csv"],
        default="mlflow",
        help="Source des données : mlflow (artifacts champion) ou csv (fichier local)",
    )
    parser.add_argument(
        "--csv-path",
        type=str,
        default=None,
        help="Chemin CSV si --source=csv (colonnes requises : label, score)",
    )
    args = parser.parse_args()

    result = main(
        target_recall=args.target_recall,
        source=args.source,
        csv_path=args.csv_path,
    )

    print("\n── Résultat ─────────────────────────────────────────")
    print(f"  threshold     : {result['threshold']:.4f}")
    print(f"  precision     : {result['precision']:.4f}")
    print(f"  recall        : {result['recall']:.4f}")
    print(f"  target_recall : {result['target_recall']:.2f}")
    print(f"  target_reached: {result['target_reached']}")
