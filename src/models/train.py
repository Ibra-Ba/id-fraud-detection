"""
Pipeline d'entraînement EfficientNet-B0 avec MLflow tracking.
Optimisé CPU — batch size réduit, epochs raisonnables sans GPU.
"""

import os
from collections.abc import Sized  #
from typing import cast

import mlflow
import mlflow.pytorch
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import TRAIN_TF, VAL_TF, IDNetDataset
from src.models.config import (
    BATCH_SIZE,
    DEVICE,
    FREEZE_EPOCHS,
    LR_FINETUNE,
    LR_HEAD,
    MIN_AUROC,
    PROCESSED_DIR,
    TOTAL_EPOCHS,
)
from src.models.efficientnet import FraudClassifier

# ─── Utilitaires de Training (Extraits pour les Tests) ──────────────────────


def check_quality_gate(auroc: float) -> bool:
    """Vérifie si l'AUROC dépasse le seuil minimal défini dans la config."""
    if np.isnan(auroc):
        raise ValueError("AUROC is NaN, quality gate cannot be evaluated.")
    return auroc >= MIN_AUROC


def make_optimizer(model: nn.Module, lr: float, phase: int) -> torch.optim.Optimizer:
    """
    Crée l'optimiseur selon la phase d'entraînement.
    Phase 1 : Uniquement la tête (classifier) est entraînée.
    Phase 2 : Tout le modèle est fine-tuné.
    """
    if phase == 1:
        # On filtre pour ne donner à l'optimiseur que ce qui n'est pas gelé
        params = [p for p in model.parameters() if p.requires_grad]
        return torch.optim.Adam(params, lr=lr)
    else:
        # Phase 2 : Tout le modèle (backbone + head)
        return torch.optim.Adam(model.parameters(), lr=lr)


# ─── Epoch Logic ──────────────────────────────────────────────────────────────


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float, float]:
    """Une epoch train ou eval. Retourne (loss, auroc, f1)."""
    training = optimizer is not None
    model.train() if training else model.eval()

    all_labels: list[int] = []
    all_probs: list[float] = []
    total_loss: float = 0.0

    with torch.set_grad_enabled(training):
        for images, labels in tqdm(loader, leave=False, desc="Training" if training else "Eval"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            logits = model(images)
            loss = criterion(logits, labels)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Calcul des probabilités pour AUROC (classe 1 = fraud)
            probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
            all_probs.extend(probs.tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
            total_loss += float(loss.item()) * len(labels)

    avg_loss = total_loss / len(all_labels)  # Plus sûr que loader.dataset
    auroc = float(roc_auc_score(all_labels, all_probs))
    preds = (np.array(all_probs) >= 0.5).astype(int)
    f1 = float(f1_score(all_labels, preds))

    return avg_loss, auroc, f1


# ─── Main Train Loop ──────────────────────────────────────────────────────────


def train() -> dict:
    # Configuration MLflow
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", "fraud-detection"))

    # Préparation Data
    train_dl = DataLoader(
        IDNetDataset(PROCESSED_DIR / "train.csv", TRAIN_TF),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )
    val_dl = DataLoader(
        IDNetDataset(PROCESSED_DIR / "val.csv", VAL_TF),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    model = FraudClassifier(pretrained=True).to(DEVICE)
    criterion = nn.CrossEntropyLoss()

    with mlflow.start_run() as run:
        # On récupère les tailles de manière sécurisée pour le linter
        train_size = len(cast(Sized, train_dl.dataset))
        val_size = len(cast(Sized, val_dl.dataset))

        mlflow.log_params(
            {
                "model": "efficientnet_b0",
                "batch_size": BATCH_SIZE,
                "freeze_epochs": FREEZE_EPOCHS,
                "total_epochs": TOTAL_EPOCHS,
                "lr_head": LR_HEAD,
                "lr_finetune": LR_FINETUNE,
                "min_auroc_required": MIN_AUROC,
                "train_samples": train_size,
                "val_samples": val_size,
            }
        )

        best_auroc: float = 0.0
        optimizer: torch.optim.Optimizer | None = None

        for epoch in range(1, TOTAL_EPOCHS + 1):
            # Gestion des phases de gel/dégel
            if epoch == 1:
                model.freeze_backbone()
                optimizer = make_optimizer(model, LR_HEAD, phase=1)
                print(f"\n--- Phase 1 : Entraînement de la tête (LR={LR_HEAD}) ---")

            elif epoch == FREEZE_EPOCHS + 1:
                model.unfreeze_backbone()
                optimizer = make_optimizer(model, LR_FINETUNE, phase=2)
                print(f"\n--- Phase 2 : Fine-tuning complet (LR={LR_FINETUNE}) ---")

            # Exécution
            tr_loss, tr_auroc, tr_f1 = run_epoch(model, train_dl, criterion, optimizer)
            vl_loss, vl_auroc, vl_f1 = run_epoch(model, val_dl, criterion)

            # Logging
            mlflow.log_metrics(
                {
                    "train_loss": tr_loss,
                    "train_auroc": tr_auroc,
                    "train_f1": tr_f1,
                    "val_loss": vl_loss,
                    "val_auroc": vl_auroc,
                    "val_f1": vl_f1,
                },
                step=epoch,
            )

            print(f"Epoch {epoch:02d} | Train AUROC: {tr_auroc:.4f} | Val AUROC: {vl_auroc:.4f}")

            # Sauvegarde du meilleur modèle
            if vl_auroc > best_auroc:
                best_auroc = vl_auroc
                mlflow.pytorch.log_model(model, artifact_path="model")
                print(f"  ⭐ Nouveau record AUROC : {best_auroc:.4f}")

        # Quality Gate finale
        passed = check_quality_gate(best_auroc)
        mlflow.log_metric("best_val_auroc", best_auroc)
        mlflow.set_tag("quality_gate", "PASSED" if passed else "FAILED")

        if not passed:
            raise ValueError(f"❌ Quality gate FAILED : {best_auroc:.4f} < {MIN_AUROC}")

        print(f"\n✅ Succès ! Meilleur AUROC : {best_auroc:.4f}")
        return {"run_id": run.info.run_id, "best_auroc": best_auroc}


if __name__ == "__main__":
    train()
