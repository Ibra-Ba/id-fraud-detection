"""
Évaluation du modèle sur le test set.
Génère : matrice de confusion, courbe ROC, rapport de classification.
Tout est loggué dans MLflow.
"""

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch
import numpy as np
import torch
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader

from src.models.train import IDNetDataset, VAL_TF, BATCH_SIZE, DEVICE

PROCESSED_DIR = Path(os.getenv("DATA_PROCESSED_DIR", "data/processed"))
MIN_AUROC = float(os.getenv("MIN_AUROC", 0.90))
MIN_F1 = float(os.getenv("MIN_F1", 0.85))


def evaluate(run_id: str) -> dict:
    """
    Évalue le modèle du run MLflow donné sur le test set.
    Retourne les métriques et logue tout dans MLflow.
    """
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

    # ── Charger le modèle depuis MLflow ──────────────────────────────────────
    model_uri = f"runs:/{run_id}/model"
    print(f"[Evaluate] Chargement modèle depuis {model_uri}")
    model = mlflow.pytorch.load_model(model_uri).to(DEVICE)
    model.eval()

    # ── Test set ──────────────────────────────────────────────────────────────
    test_dl = DataLoader(
        IDNetDataset(PROCESSED_DIR / "test.csv", VAL_TF),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    all_labels, all_probs = [], []
    with torch.no_grad():
        for images, labels in test_dl:
            probs = torch.softmax(model(images), dim=1)[:, 1].numpy()
            all_probs.extend(probs)
            all_labels.extend(labels.numpy())

    all_preds = (np.array(all_probs) >= 0.5).astype(int)
    auroc = roc_auc_score(all_labels, all_probs)
    f1 = f1_score(all_labels, all_preds)

    # ── Rapport console ───────────────────────────────────────────────────────
    print("\n" + "─" * 50)
    print(classification_report(all_labels, all_preds, target_names=["genuine", "fraud"]))
    print(f"AUROC : {auroc:.4f}  |  F1 : {f1:.4f}")
    print("─" * 50)

    # ── Log MLflow ────────────────────────────────────────────────────────────
    with mlflow.start_run(run_id=run_id):
        mlflow.log_metrics({"test_auroc": auroc, "test_f1": f1})

        # Matrice de confusion
        cm = confusion_matrix(all_labels, all_preds)
        fig, ax = plt.subplots(figsize=(6, 5))
        ConfusionMatrixDisplay(cm, display_labels=["genuine", "fraud"]).plot(ax=ax)
        ax.set_title("Confusion Matrix — Test Set")
        mlflow.log_figure(fig, "confusion_matrix.png")
        plt.close()

        # Courbe ROC
        fig, ax = plt.subplots(figsize=(6, 5))
        RocCurveDisplay.from_predictions(all_labels, all_probs, ax=ax)
        ax.set_title(f"ROC Curve — AUROC={auroc:.4f}")
        mlflow.log_figure(fig, "roc_curve.png")
        plt.close()

        # Quality gate final
        passed = auroc >= MIN_AUROC and f1 >= MIN_F1
        mlflow.set_tag("test_quality_gate", "PASSED" if passed else "FAILED")

    # ── Quality gate ──────────────────────────────────────────────────────────
    if not passed:
        raise ValueError(
            f"❌ Quality gate FAILED : "
            f"auroc={auroc:.4f} (min={MIN_AUROC}) | "
            f"f1={f1:.4f} (min={MIN_F1})"
        )

    print(f"\n✅ Évaluation réussie — AUROC={auroc:.4f} | F1={f1:.4f}")
    return {"run_id": run_id, "test_auroc": auroc, "test_f1": f1}


if __name__ == "__main__":
    run_id = sys.argv[1] if len(sys.argv) > 1 else os.getenv("MLFLOW_RUN_ID")
    if not run_id:
        raise ValueError("Fournir run_id en argument ou via MLFLOW_RUN_ID")
    evaluate(run_id)
