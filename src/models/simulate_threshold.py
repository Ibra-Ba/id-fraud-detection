"""
Simulation de l'impact du seuil sur recall/precision.

Usage:
    python -m src.models.simulate_threshold
    python -m src.models.simulate_threshold --target-recall 0.95
"""

import argparse
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
from sklearn.metrics import classification_report, precision_recall_curve
from torch.utils.data import DataLoader

from src.data.dataset import VAL_TF, IDNetDataset
from src.models.config import BATCH_SIZE, DEVICE, PROCESSED_DIR
from src.models.efficientnet import FraudClassifier

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_scores(csv_path: Path, source="mlflow") -> tuple[np.ndarray, np.ndarray]:
    """Charge le modèle (local ou MLflow) et calcule les scores."""

    import mlflow.pytorch

    # ── Load model ─────────────────────────────────────────────

    if source == "local":
        checkpoint = Path("best_model_checkpoint.pt")
        if not checkpoint.exists():
            raise FileNotFoundError("Checkpoint local introuvable")

        print("[INFO] Loading model from local checkpoint")
        model = FraudClassifier(pretrained=False)
        model.load_state_dict(torch.load(str(checkpoint), map_location=DEVICE))

    else:
        print("[INFO] Loading model from MLflow (champion)")

        uri = os.getenv("MLFLOW_TRACKING_URI")
        if uri:
            mlflow.set_tracking_uri(uri)

        model_name = os.getenv("MLFLOW_MODEL_NAME", "IDNet-Fraud-Detector")
        model = mlflow.pytorch.load_model(f"models:/{model_name}@champion")

    model = model.to(DEVICE)
    model.eval()

    # ── Data ───────────────────────────────────────────────────
    loader = DataLoader(
        IDNetDataset(csv_path, VAL_TF),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    # ── Inference ──────────────────────────────────────────────
    all_probs, all_labels = [], []

    with torch.no_grad():
        for images, labels in loader:
            probs = torch.softmax(model(images.to(DEVICE)), dim=1)[:, 1]
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.numpy())

    return np.array(all_labels), np.array(all_probs)


def find_threshold_for_recall(
    y_true: np.ndarray,
    y_score: np.ndarray,
    target_recall: float = 0.90,
) -> float:
    """
    Trouve le seuil le plus élevé qui garantit recall >= target_recall.
    Maximise la precision sous contrainte de recall minimum.
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_score)

    valid = [
        (p, r, t)
        for p, r, t in zip(precisions, recalls, thresholds)  # noqa: B905
        if r >= target_recall
    ]

    if not valid:
        logger.warning(
            f"Impossible d'atteindre recall={target_recall:.2f} — " "utilise le seuil minimal"
        )
        return float(thresholds[0])

    # Parmi les seuils valides, prend celui avec la meilleure precision
    best = max(valid, key=lambda x: x[0])
    logger.info(
        f"Seuil optimal pour recall>={target_recall:.2f} : "
        f"{best[2]:.4f} (precision={best[0]:.4f}, recall={best[1]:.4f})"
    )
    return float(best[2])


def simulate(
    y_true: np.ndarray,
    y_score: np.ndarray,
    thresholds: list[float],
) -> pd.DataFrame:
    """Simule les métriques pour chaque seuil."""
    rows = []
    for t in thresholds:
        preds = (y_score >= t).astype(int)
        report = classification_report(
            y_true,
            preds,
            target_names=["genuine", "fraud"],
            output_dict=True,
            zero_division=0,
        )

        # Calculer les compteurs brute
        # (y_true == 1) & (preds == 0) crée un tableau de Booléens
        # .sum() compte les True
        n_missed = np.sum((y_true == 1) & (preds == 0))
        n_alarms = np.sum((y_true == 0) & (preds == 1))

        rows.append(
            {
                "threshold": round(t, 4),
                "fraud_recall": round(report["fraud"]["recall"], 4),  # type: ignore
                "fraud_precision": round(report["fraud"]["precision"], 4),  # type: ignore
                "fraud_f1": round(report["fraud"]["f1-score"], 4),  # type: ignore
                "genuine_recall": round(report["genuine"]["recall"], 4),  # type: ignore
                "accuracy": round(report["accuracy"], 4),  # type: ignore
                "n_fraud_missed": int(n_missed),
                "n_false_alarms": int(n_alarms),
            }
        )
    return pd.DataFrame(rows)


def main(target_recall: float = 0.90, csv_path: Path | None = None, source="mlflow"):  # noqa: UP045
    csv_path = csv_path or (PROCESSED_DIR / "test.csv")

    logger.info(f"Chargement des scores depuis {csv_path}...")
    y_true, y_score = load_scores(csv_path, source=source)
    logger.info(f"{len(y_true)} échantillons | fraud_rate={y_true.mean():.2%}")

    # Grille de seuils à tester
    # thresholds = [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.9262]
    # Debug
    # thresholds = [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
    thresholds = np.linspace(0.0, 1.0, 50).tolist()

    # Seuil optimal pour le recall cible
    optimal = find_threshold_for_recall(y_true, y_score, target_recall)
    if optimal not in thresholds:
        thresholds.append(optimal)  # type: ignore
    thresholds = sorted(thresholds)

    # Simulation
    df = simulate(y_true, y_score, thresholds)

    print("\n─── Simulation seuils ───────────────────────────────────────────")
    print(df.to_string(index=False))

    print(f"\n─── Recommandation (recall fraud >= {target_recall:.0%}) ───────")
    rec = df[df["fraud_recall"] >= target_recall]
    if rec.empty:
        print("⚠️  Aucun seuil n'atteint le recall cible — baisse target_recall")
    else:
        best = rec.loc[rec["fraud_precision"].idxmax()]
        print(f"  Seuil recommandé  : {best['threshold']}")
        print(f"  Fraud recall      : {best['fraud_recall']:.2%}")
        print(f"  Fraud precision   : {best['fraud_precision']:.2%}")
        print(f"  Fraudes manquées  : {best['n_fraud_missed']}")
        print(f"  Fausses alertes   : {best['n_false_alarms']}")
        print(f"  Accuracy          : {best['accuracy']:.2%}")
        print(f"\n  → Met à jour optimal_threshold={best['threshold']} dans MLflow")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--target-recall",
        type=float,
        default=0.90,
        help="Recall fraud minimum souhaité (défaut: 0.95)",
    )

    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="CSV à évaluer (défaut: test.csv)",
    )

    parser.add_argument(
        "--source",
        type=str,
        choices=["mlflow", "local"],
        default="mlflow",
        help="Source du modèle (mlflow ou local). Défaut: mlflow",
    )

    args = parser.parse_args()

    main(
        target_recall=args.target_recall,
        csv_path=Path(args.csv) if args.csv else None,
        source=args.source,
    )
