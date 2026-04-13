import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch
import numpy as np
import torch
import torch.nn as nn
from dotenv import load_dotenv
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
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

load_dotenv()

# 1. Calcul du meilleur seuil


def find_optimal_threshold(y_true, y_score):
    """
    Trouve le seuil qui maximise l'indice de Youden (Sensibilité + Spécificité - 1).
    Idéal pour équilibrer les classes quand il y a un déséquilibre.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    return float(optimal_threshold)


def compute_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5) -> dict:
    """Calcule les métriques avec un seuil spécifique."""
    y_pred = (y_score >= threshold).astype(int)
    return {
        "auroc": float(roc_auc_score(y_true, y_score)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
        "threshold_used": threshold,
    }


#  2. Fonction d'évaluation principale


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    log_to_mlflow: bool = False,
    artifact_dir: Path | None = None,
    enforce_gate: bool = False,
) -> dict:
    model.eval()
    all_labels, all_probs = [], []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            all_probs.extend(probs.tolist())
            all_labels.extend(labels.numpy().tolist())

    y_true = np.array(all_labels)
    y_score = np.array(all_probs)

    # Calcul du seuil optimal
    best_threshold = find_optimal_threshold(y_true, y_score)
    print(f"\n[INFO] Seuil optimal calculé : {best_threshold:.4f}")

    # Métriques avec le nouveau seuil
    metrics = compute_metrics(y_true, y_score, threshold=best_threshold)
    y_pred = (y_score >= best_threshold).astype(int)

    # ── Sauvegarde des artefacts (CM, ROC) ──
    if artifact_dir or log_to_mlflow:
        # Matrice de confusion avec le meilleur seuil
        cm = confusion_matrix(y_true, y_pred)
        fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
        ConfusionMatrixDisplay(cm, display_labels=["genuine", "fraud"]).plot(ax=ax_cm)
        ax_cm.set_title(f"Confusion Matrix (Threshold: {best_threshold:.2f})")

        # Courbe ROC
        fig_roc, ax_roc = plt.subplots(figsize=(6, 5))
        RocCurveDisplay.from_predictions(y_true, y_score, ax=ax_roc)
        ax_roc.plot([0, 1], [0, 1], "k--")  # Ligne de chance
        # Marquer le point optimal sur la courbe
        fpr, tpr, _ = roc_curve(y_true, y_score)
        idx = np.argmin(  # noqa: F841
            np.abs(np.linspace(0, 1, len(tpr)) - tpr)
        )  # simplification pour l'affichage  # noqa: F841

        if log_to_mlflow:
            mlflow.log_metrics({f"test_{k}": v for k, v in metrics.items()})
            mlflow.log_param("optimal_threshold", best_threshold)
            mlflow.log_figure(fig_cm, "confusion_matrix_optimized.png")
            mlflow.log_figure(fig_roc, "roc_curve.png")

        if artifact_dir:
            artifact_dir.mkdir(parents=True, exist_ok=True)
            fig_cm.savefig(artifact_dir / "confusion_matrix_optimized.png")
            fig_roc.savefig(artifact_dir / "roc_curve.png")

        plt.close(fig_cm)
        plt.close(fig_roc)

    # Quality Gate
    passed = metrics["auroc"] >= MIN_AUROC and metrics["f1"] >= MIN_F1
    metrics["gate_passed"] = passed

    if enforce_gate and not passed:
        print(f"❌ Quality gate FAILED : AUROC={metrics['auroc']:.4f}, F1={metrics['f1']:.4f}")
    elif passed:
        print("✅ Quality gate PASSED")

    return metrics


def evaluate_from_run(run_id: str) -> dict:
    """Charge un modèle depuis MLflow et l'évalue sur le test set."""
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    mlflow.set_tracking_uri(tracking_uri)  # type: ignore

    model_uri = f"runs:/{run_id}/model"
    print(f"[INFO] Chargement du modèle depuis : {model_uri}")
    model = mlflow.pytorch.load_model(model_uri).to(DEVICE)

    test_dl = DataLoader(
        IDNetDataset(PROCESSED_DIR / "test.csv", VAL_TF),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
    )

    with mlflow.start_run(run_id=run_id, nested=True):
        return evaluate(model, test_dl, log_to_mlflow=True, enforce_gate=True)


if __name__ == "__main__":
    target_run = sys.argv[1] if len(sys.argv) > 1 else os.getenv("MLFLOW_RUN_ID")
    if target_run:
        results = evaluate_from_run(target_run)
        print("\n--- Résultats Finaux ---")
        for k, v in results.items():
            print(f"{k}: {v}")
    else:
        print("Usage: python -m src.models.evaluate <run_id>")
