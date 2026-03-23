import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader

from src.data.dataset import VAL_TF, IDNetDataset
from src.models.config import (
    BATCH_SIZE,
    DEVICE,
    MIN_AUROC,
    MIN_F1,
    PROCESSED_DIR,
)

# ─── 1. Calcul des métriques (Utilisé par les tests) ───────────────────────


def compute_metrics(y_true: np.ndarray, y_score: np.ndarray) -> dict:
    """Calcule l'AUROC, l'Accuracy et le F1-score."""
    y_pred = (y_score >= 0.5).astype(int)
    return {
        "auroc": float(roc_auc_score(y_true, y_score)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
    }


# ─── 2. Fonction d'évaluation principale (Modulaire) ────────────────────────


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    log_to_mlflow: bool = False,
    artifact_dir: Path | None = None,
    enforce_gate: bool = False,
) -> dict:
    """
    Évalue un modèle sur un loader donné.
    Peut logger sur MLflow ou sauvegarder des artefacts localement.
    """
    model.eval()
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            all_probs.extend(probs.tolist())
            all_labels.extend(labels.numpy().tolist())

    y_true = np.array(all_labels)
    y_score = np.array(all_probs)
    metrics = compute_metrics(y_true, y_score)

    # ── Sauvegarde des artefacts (CM, ROC) ──
    if artifact_dir or log_to_mlflow:
        # Matrice de confusion
        y_pred = (y_score >= 0.5).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
        ConfusionMatrixDisplay(cm, display_labels=["genuine", "fraud"]).plot(ax=ax_cm)

        # Courbe ROC
        fig_roc, ax_roc = plt.subplots(figsize=(6, 5))
        RocCurveDisplay.from_predictions(y_true, y_score, ax=ax_roc)

        if log_to_mlflow:
            mlflow.log_metrics({f"test_{k}": v for k, v in metrics.items()})
            mlflow.log_figure(fig_cm, "confusion_matrix.png")
            mlflow.log_figure(fig_roc, "roc_curve.png")

        if artifact_dir:
            artifact_dir.mkdir(parents=True, exist_ok=True)
            fig_cm.savefig(artifact_dir / "confusion_matrix.png")
            fig_roc.savefig(artifact_dir / "roc_curve.png")

        plt.close(fig_cm)
        plt.close(fig_roc)

    # ── Quality Gate ──
    passed = metrics["auroc"] >= MIN_AUROC and metrics["f1"] >= MIN_F1
    metrics["gate_passed"] = passed

    if enforce_gate and not passed:
        raise ValueError(f"❌ Quality gate FAILED : AUROC={metrics['auroc']:.4f}")

    return metrics


# ─── 3. Point d'entrée pour script (via run_id) ───────────────────────────


def evaluate_from_run(run_id: str) -> dict:
    """Charge un modèle depuis MLflow et l'évalue sur le test set."""
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)

    model_uri = f"runs:/{run_id}/model"
    model = mlflow.pytorch.load_model(model_uri).to(DEVICE)

    test_dl = DataLoader(
        IDNetDataset(PROCESSED_DIR / "test.csv", VAL_TF), batch_size=BATCH_SIZE, shuffle=False
    )

    with mlflow.start_run(run_id=run_id):
        return evaluate(model, test_dl, log_to_mlflow=True, enforce_gate=True)


if __name__ == "__main__":
    target_run = sys.argv[1] if len(sys.argv) > 1 else os.getenv("MLFLOW_RUN_ID")
    if target_run:
        evaluate_from_run(target_run)
    else:
        print("Usage: python -m src.models.evaluate <run_id>")
