import logging
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

from src.models.simulate_threshold import find_threshold_for_recall

# 1. Chargement des variables d'environnement
load_dotenv()

MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "IDNet-Fraud-Detector")

# Logs pour debug


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

logging.getLogger("mlflow").setLevel(logging.DEBUG)
logging.getLogger("urllib3").setLevel(logging.DEBUG)
logging.getLogger("requests").setLevel(logging.DEBUG)


def get_model_scores(model, loader):
    model.eval()
    all_probs, all_labels = [], []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            probs = torch.softmax(model(images), dim=1)[:, 1]

            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.numpy())

    return np.array(all_labels), np.array(all_probs)


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

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")

    if not tracking_uri:
        raise RuntimeError("MLFLOW_TRACKING_URI non défini")

    mlflow.set_tracking_uri(tracking_uri)

    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "fraud-detection")
    mlflow.set_experiment(experiment_name)

    # ─── Hard reset du contexte MLflow ───────────────────

    if mlflow.active_run() is not None:
        print("Active MLflow run detected → closing it")
        mlflow.end_run()

    os.environ.pop("MLFLOW_RUN_ID", None)

    # Optimisation CPU : On détecte le nombre de cœurs disponibles

    cpus = os.cpu_count() or 2
    workers = min(4, cpus)

    train_dl = DataLoader(
        IDNetDataset(PROCESSED_DIR / "train.csv", TRAIN_TF),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
    )

    val_dl = DataLoader(
        IDNetDataset(PROCESSED_DIR / "val.csv", VAL_TF),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
    )

    model = FraudClassifier(pretrained=True).to(DEVICE)
    # compute class counts une fois
    class_counts = np.bincount(IDNetDataset(PROCESSED_DIR / "train.csv", TRAIN_TF).df["label"])
    freq = class_counts / class_counts.sum()

    weights = 1.0 / np.sqrt(freq)
    weights = torch.tensor(weights, dtype=torch.float32).to(DEVICE)

    criterion = nn.CrossEntropyLoss(weight=weights)
    best_auroc = 0.0

    print(f"\n[START] Entraînement sur {DEVICE} | Workers: {workers}")
    print(f"[INFO] Tracking vers: {tracking_uri}\n")
    run_id = None
    try:
        with mlflow.start_run(run_name="training") as run:
            run_id = run.info.run_id

            # tags contextuels
            mlflow.set_tag("retrain_reason", os.getenv("REASON", "manual"))
            mlflow.set_tag("pipeline", "training")
            mlflow.set_tag("trigger", "github_actions")

            mlflow.log_params({"model": "efficientnet_b0", "batch_size": BATCH_SIZE})

            # Early Stopping
            patience = 3
            patience_counter = 0
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
                    # Phase 2 : dégel un peu plus profond
                    for name, param in model.backbone.named_parameters():
                        param.requires_grad = any(
                            x in name
                            for x in ["blocks.4", "blocks.5", "blocks.6", "blocks.7", "classifier"]
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

                # --- SAUVEGARDE + EARLY STOPPING ---

                if vl_auroc > best_auroc:
                    best_auroc = vl_auroc
                    torch.save(model.state_dict(), "best_model_checkpoint.pt")
                    patience_counter = 0  # reset
                else:
                    patience_counter += 1
                    print(f"⏳ Early stopping patience: {patience_counter}/{patience}")

                    if patience_counter >= patience:
                        print("⏹ Early stopping déclenché")
                        break

            print("\n--- Calcul du seuil optimal (Target Recall: 95%) ---")

            # On enregistre dans MLflow

            best_model = FraudClassifier(pretrained=False).to(DEVICE)
            best_model.load_state_dict(torch.load("best_model_checkpoint.pt"))
            best_model.eval()

            #  Calcul LOCAL du threshold
            y_true, y_score = get_model_scores(best_model, val_dl)

            optimal_threshold = find_threshold_for_recall(
                y_true,
                y_score,
                target_recall=0.95,
            )

            # Print for debugging
            # 🔍 DEBUG distribution des scores
            fraud_scores = y_score[y_true == 1]
            genuine_scores = y_score[y_true == 0]

            print("\n--- DEBUG SCORES DISTRIBUTION ---")
            print(f"Fraud mean     : {fraud_scores.mean():.4f}")
            print(f"Fraud min/max  : {fraud_scores.min():.4f} / {fraud_scores.max():.4f}")
            print(f"Genuine mean   : {genuine_scores.mean():.4f}")
            print(f"Genuine min/max: {genuine_scores.min():.4f} / {genuine_scores.max():.4f}")

            print(f"Percentile 90 fraud : {np.percentile(fraud_scores, 90):.4f}")
            print(f"Percentile 10 fraud : {np.percentile(fraud_scores, 10):.4f}")

            # Logging MLflow
            mlflow.pytorch.log_model(
                best_model,
                artifact_path="model",
                pip_requirements="requirements.txt",
            )

            mlflow.log_artifact("best_model_checkpoint.pt", artifact_path="model")

            mlflow.log_metric("optimal_threshold", optimal_threshold)
            mlflow.log_metric("best_val_auroc", best_auroc)
            mlflow.log_metric("max_recall_possible", y_score[y_true == 1].min())
            mlflow.set_tag("optimal_threshold", str(optimal_threshold))
            print("✅ Modèle loggé avec code embarqué")
            print(f"✅ Seuil optimal enregistré : {optimal_threshold}", flush=True)

    except KeyboardInterrupt:
        print(
            "\n[STOP] Interruption. Le fichier 'best_model_checkpoint.pt' contient votre meilleur modèle."
        )
    finally:
        mlflow.end_run()
    print("TRACKING URI =", mlflow.get_tracking_uri())
    return run_id


if __name__ == "__main__":
    train()
