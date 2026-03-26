import os

import mlflow
import mlflow.pytorch
import numpy as np
import torch
import torch.nn as nn
from dotenv import load_dotenv
from sklearn.metrics import f1_score, roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

# 1. Chargement des variables d'environnement
load_dotenv()

from src.data.dataset import TRAIN_TF, VAL_TF, IDNetDataset  # noqa: E402
from src.models.config import (  # noqa: E402
    BATCH_SIZE,
    DEVICE,
    FREEZE_EPOCHS,
    LR_FINETUNE,
    LR_HEAD,
    MIN_AUROC,
    PROCESSED_DIR,
    TOTAL_EPOCHS,
)
from src.models.efficientnet import FraudClassifier  # noqa: E402


def check_quality_gate(auroc: float) -> bool:
    if auroc != auroc:  # NaN check
        raise ValueError("AUROC is NaN")
    return auroc >= MIN_AUROC


def make_optimizer(model, lr: float, phase: int):
    params = filter(lambda p: p.requires_grad, model.parameters())
    return torch.optim.Adam(params, lr=lr)


def run_epoch(model, loader, criterion, optimizer=None):
    """Exécute une époque et affiche la progression en temps réel."""
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    all_labels, all_probs = [], []
    total_loss = 0.0

    # Barre de progression dynamique
    desc = "🚀 Train" if is_train else "🧪 Eval "
    pbar = tqdm(loader, desc=desc, leave=False, unit="batch")

    with torch.set_grad_enabled(is_train):
        for images, labels in pbar:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            logits = model(images)
            loss = criterion(logits, labels)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Calcul des probabilités pour AUROC
            probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
            all_probs.extend(probs.tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
            total_loss += loss.item() * len(labels)

            # Mise à jour de l'affichage (Loss moyenne en direct)
            pbar.set_postfix({"loss": f"{total_loss/len(all_labels):.3f}"})

    avg_loss = total_loss / len(all_labels)
    auroc = float(roc_auc_score(all_labels, all_probs))
    f1 = float(f1_score(all_labels, (np.array(all_probs) >= 0.5).astype(int)))

    return avg_loss, auroc, f1


def train():
    # Configuration MLflow
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", "fraud-detection"))

    # Optimisation CPU : On détecte le nombre de cœurs disponibles
    cpus = os.cpu_count() or 2
    workers = min(4, cpus)

    train_dl = DataLoader(
        IDNetDataset(PROCESSED_DIR / "train.csv", TRAIN_TF),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=True,
    )
    val_dl = DataLoader(
        IDNetDataset(PROCESSED_DIR / "val.csv", VAL_TF),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
    )

    model = FraudClassifier(pretrained=True).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    best_auroc = 0.0

    print(f"\n[START] Entraînement sur {DEVICE} | Workers: {workers}")
    print(f"[INFO] Tracking vers: {tracking_uri}\n")

    try:
        with mlflow.start_run():
            mlflow.log_params({"model": "efficientnet_b0", "batch_size": BATCH_SIZE})

            for epoch in range(1, TOTAL_EPOCHS + 1):
                # --- GESTION DES PHASES ---
                if epoch == 1:
                    model.freeze_backbone()
                    optimizer = torch.optim.Adam(
                        filter(lambda p: p.requires_grad, model.parameters()), lr=LR_HEAD
                    )
                    print(f"--- Phase 1: Tête (LR={LR_HEAD}) ---")

                elif epoch == FREEZE_EPOCHS + 1:
                    print(f"\n--- Phase 2: Fine-tuning Partiel (LR={LR_FINETUNE}) ---")
                    # OPTIMISATION : On ne dégel que les blocs 6 et 7 pour la vitesse CPU
                    for name, param in model.backbone.named_parameters():
                        param.requires_grad = any(
                            x in name for x in ["blocks.6", "blocks.7", "classifier"]
                        )
                    optimizer = torch.optim.Adam(
                        filter(lambda p: p.requires_grad, model.parameters()), lr=LR_FINETUNE
                    )

                # --- RUN ---
                tr_loss, tr_auroc, _ = run_epoch(model, train_dl, criterion, optimizer)
                vl_loss, vl_auroc, vl_f1 = run_epoch(model, val_dl, criterion)

                # --- LOGGING ---
                mlflow.log_metrics(
                    {"tr_loss": tr_loss, "vl_loss": vl_loss, "vl_auroc": vl_auroc, "vl_f1": vl_f1},
                    step=epoch,
                )

                print(
                    f"Epoch {epoch:02d} | Train AUROC: {tr_auroc:.4f} | Val AUROC: {vl_auroc:.4f}"
                )

                # --- SAUVEGARDE SÉCURISÉE ---
                if vl_auroc > best_auroc:
                    best_auroc = vl_auroc
                    # 1. Sauvegarde locale (survit au crash réseau)
                    torch.save(model.state_dict(), "best_model_checkpoint.pt")
                    # 2. MLflow
                    try:
                        mlflow.pytorch.log_model(model, artifact_path="model")
                        print(f"  ⭐ Record Sauvegardé : {best_auroc:.4f}")
                    except Exception as e:
                        print(f"  ⚠️ Erreur upload MLflow (Backup local OK) : {e}")

    except KeyboardInterrupt:
        print(
            "\n[STOP] Interruption. Le fichier 'best_model_checkpoint.pt' contient votre meilleur modèle."
        )
    finally:
        mlflow.end_run()


if __name__ == "__main__":
    train()
