"""
Pipeline d'entraînement EfficientNet-B0 avec MLflow tracking.
Optimisé CPU — batch size réduit, epochs raisonnables sans GPU.
"""

import os

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
    IMG_SIZE,
    LR_FINETUNE,
    LR_HEAD,
    MIN_AUROC,
    PROCESSED_DIR,
    TOTAL_EPOCHS,
)
from src.models.efficientnet import FraudClassifier

# ─── Transforms ───────────────────────────────────────────────────────────────


# ─── Dataset ──────────────────────────────────────────────────────────────────


# ─── Epoch ────────────────────────────────────────────────────────────────────
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
        for images, labels in tqdm(loader, leave=False):
            logits = model(images)
            loss = criterion(logits, labels)

            if training:
                optimizer.zero_grad()  # type: ignore[union-attr]
                loss.backward()
                optimizer.step()  # type: ignore[union-attr]

            probs = torch.softmax(logits, dim=1)[:, 1].detach().numpy()
            all_probs.extend(probs.tolist())  # float natif
            all_labels.extend(labels.numpy().tolist())
            total_loss += float(loss.item()) * len(labels)

    avg_loss = total_loss / len(loader.dataset)  # type: ignore[arg-type]
    auroc = float(roc_auc_score(all_labels, all_probs))
    preds = (np.array(all_probs) >= 0.5).astype(int)
    f1 = float(f1_score(all_labels, preds))
    return avg_loss, auroc, f1


# ─── Train ────────────────────────────────────────────────────────────────────
def train() -> dict:
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", "fraud-detection"))

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
        mlflow.log_params(
            {
                "model": "efficientnet_b0",
                "batch_size": BATCH_SIZE,
                "freeze_epochs": FREEZE_EPOCHS,
                "total_epochs": TOTAL_EPOCHS,
                "lr_head": LR_HEAD,
                "lr_finetune": LR_FINETUNE,
                "device": DEVICE,
                "img_size": IMG_SIZE[0],
                "train_samples": len(train_dl.dataset),  # type: ignore[arg-type]
                "val_samples": len(val_dl.dataset),  # type: ignore[arg-type]
            }
        )

        best_auroc: float = 0.0
        optimizer: torch.optim.Optimizer | None = None

        for epoch in range(1, TOTAL_EPOCHS + 1):

            if epoch == 1:
                model.freeze_backbone()
                optimizer = torch.optim.Adam(model.head.parameters(), lr=LR_HEAD)
            elif epoch == FREEZE_EPOCHS + 1:
                model.unfreeze_backbone()
                optimizer = torch.optim.Adam(model.parameters(), lr=LR_FINETUNE)

            tr_loss, tr_auroc, tr_f1 = run_epoch(model, train_dl, criterion, optimizer)
            vl_loss, vl_auroc, vl_f1 = run_epoch(model, val_dl, criterion)

            # cast explicite en float pour MLflow
            mlflow.log_metrics(
                {
                    "train_loss": float(tr_loss),
                    "train_auroc": float(tr_auroc),
                    "train_f1": float(tr_f1),
                    "val_loss": float(vl_loss),
                    "val_auroc": float(vl_auroc),
                    "val_f1": float(vl_f1),
                },
                step=epoch,
            )

            print(
                f"Epoch {epoch:02d}/{TOTAL_EPOCHS} | "
                f"train_auroc={tr_auroc:.4f} | "
                f"val_auroc={vl_auroc:.4f} | "
                f"val_f1={vl_f1:.4f}"
            )

            if vl_auroc > best_auroc:
                best_auroc = vl_auroc
                mlflow.pytorch.log_model(model, artifact_path="model")
                print(f"  ✅ Meilleur modèle sauvegardé (auroc={best_auroc:.4f})")

        mlflow.log_metric("best_val_auroc", float(best_auroc))
        passed = best_auroc >= MIN_AUROC
        mlflow.set_tag("quality_gate", "PASSED" if passed else "FAILED")
        mlflow.set_tag("model_version", "efficientnet_b0")

        if not passed:
            raise ValueError(f"❌ Quality gate FAILED : best_auroc={best_auroc:.4f} < {MIN_AUROC}")

        print(f"\n✅ Entraînement terminé — meilleur AUROC : {best_auroc:.4f}")
        return {"run_id": run.info.run_id, "best_auroc": best_auroc}


if __name__ == "__main__":
    result = train()
    print(f"run_id : {result['run_id']}")
