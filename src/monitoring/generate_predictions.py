"""
Génération des prédictions — IDNet Fraud Detector Bac+5

Charge le modèle champion depuis MLflow, génère les prédictions
sur un CSV manifest (image_path + label), et pousse les résultats
sur S3 sous forme de fichiers JSON (un par image).

Ces fichiers JSON sont ensuite consommés par monitor_pro.py via
load_s3_predictions().

Structure JSON produite (cohérente avec monitor_pro.py) :
    {
        "id": "ESP_genuine_001",
        "label": 0,
        "score": 0.0732,
        "prediction": 0,
        "timestamp": "2025-06-07T06:00:00Z"
    }

Usage:
    python -m src.monitoring.generate_predictions
    python -m src.monitoring.generate_predictions \\
        --csv-path data/processed/val.csv \\
        --prefix predictions/val_run \\
        --max-samples 200
"""

import argparse
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

import boto3
import mlflow
import mlflow.pytorch
import torch
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient
from torch.utils.data import DataLoader

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Constantes ────────────────────────────────────────────────────────────────
MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "IDNet-Fraud-Detector")
S3_BUCKET = os.getenv("S3_BUCKET")
DEVICE = "cpu"
NUM_WORKERS = 0
IMAGE_SIZE = 224
BATCH_SIZE = 32


# ── Chargement du modèle ──────────────────────────────────────────────────────


def load_champion_model():
    """Charge le modèle @champion depuis MLflow."""
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    client = MlflowClient()

    try:
        mv = client.get_model_version_by_alias(MODEL_NAME, "champion")
    except Exception as e:
        raise RuntimeError(f"Champion introuvable : {e}") from e

    model_uri = f"models:/{MODEL_NAME}@champion"
    logger.info(f"Chargement du modèle {MODEL_NAME} v{mv.version} depuis {model_uri}")

    model = mlflow.pytorch.load_model(model_uri, map_location=DEVICE)
    model.eval()

    # Threshold : metrics → tags (cohérent avec register_model.py Bac+4)
    run = client.get_run(mv.run_id)
    threshold = run.data.metrics.get("optimal_threshold")
    if threshold is None:
        threshold = float(run.data.tags.get("optimal_threshold", "0.5"))

    logger.info(f"Modèle chargé — threshold={threshold:.4f}")
    return model, float(threshold), mv.version


# ── Dataset minimal ───────────────────────────────────────────────────────────


class PredictionDataset(torch.utils.data.Dataset):
    """Dataset léger pour l'inférence — charge les images depuis image_path."""

    def __init__(self, df, transform=None):
        import torchvision.transforms as T

        self.df = df.reset_index(drop=True)
        self.transform = transform or T.Compose(
            [
                T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        from PIL import Image

        row = self.df.iloc[idx]
        img_path = row["image_path"]

        try:
            img = Image.open(img_path).convert("RGB")
            tensor = self.transform(img)
        except Exception as e:
            logger.warning(f"Image illisible {img_path} : {e} — zéros utilisés")
            tensor = torch.zeros(3, IMAGE_SIZE, IMAGE_SIZE)

        label = int(row["label"]) if "label" in row else -1
        sample_id = Path(img_path).stem
        return tensor, label, sample_id


# ── Inférence ─────────────────────────────────────────────────────────────────


def run_inference(model, df, threshold: float, max_samples: int | None = None) -> list[dict]:
    """
    Génère les prédictions sur le DataFrame.
    Retourne une liste de dicts prêts à être sérialisés en JSON.
    """
    if max_samples and len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=42)
        logger.info(f"Sous-échantillonnage : {max_samples} images")

    dataset = PredictionDataset(df)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    records = []
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    with torch.no_grad():
        for images, labels, sample_ids in loader:
            images = images.to(DEVICE)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()

            for i, (score, sample_id) in enumerate(zip(probs, sample_ids, strict=False)):
                score_f = float(score)
                label = int(labels[i])
                records.append(
                    {
                        "id": sample_id,
                        "label": label if label >= 0 else None,
                        "score": round(score_f, 6),
                        "prediction": int(score_f >= threshold),
                        "timestamp": timestamp,
                    }
                )

    logger.info(f"{len(records)} prédictions générées")
    return records


# ── Push S3 ───────────────────────────────────────────────────────────────────


def push_to_s3(records: list[dict], prefix: str) -> int:
    """
    Pousse les records sur S3.
    Chaque record → un fichier JSON : s3://{bucket}/{prefix}/{id}.json
    Retourne le nombre de fichiers uploadés.
    """
    if not S3_BUCKET:
        raise RuntimeError("S3_BUCKET non défini dans l'environnement")

    s3 = boto3.client("s3")
    uploaded = 0

    for record in records:
        key = f"{prefix}/{record['id']}.json"
        body = json.dumps(record, ensure_ascii=False)

        s3.put_object(
            Bucket=S3_BUCKET,
            Key=key,
            Body=body.encode("utf-8"),
            ContentType="application/json",
        )
        uploaded += 1

    logger.info(f"{uploaded} fichiers uploadés sur s3://{S3_BUCKET}/{prefix}/")
    return uploaded


# ── Point d'entrée principal ──────────────────────────────────────────────────


def run(
    csv_path: str,
    prefix: str,
    max_samples: int | None = None,
) -> int:
    """
    Pipeline complet : charge modèle → inférence → push S3.

    Args:
        csv_path: chemin vers le CSV manifest (colonnes: image_path, label)
        prefix: préfixe S3 de destination (ex: predictions/train_ref)
        max_samples: limite le nombre d'images (None = tout)

    Returns:
        nombre de fichiers uploadés sur S3
    """
    import pandas as pd

    logger.info(f"── generate_predictions : {csv_path} → s3://{S3_BUCKET}/{prefix}/")

    # Charge manifest
    csv = Path(csv_path)
    if not csv.exists():
        raise FileNotFoundError(f"CSV manifest introuvable : {csv}")

    df = pd.read_csv(csv)
    logger.info(f"Manifest chargé : {len(df)} entrées")

    if "image_path" not in df.columns:
        raise ValueError(
            f"Colonne 'image_path' manquante dans {csv_path}. "
            f"Colonnes disponibles : {list(df.columns)}"
        )

    # Charge modèle
    model, threshold, version = load_champion_model()

    # Inférence
    records = run_inference(model, df, threshold, max_samples)

    # Push S3
    uploaded = push_to_s3(records, prefix)

    logger.info(f"Done — {uploaded} prédictions dans s3://{S3_BUCKET}/{prefix}/")
    return uploaded


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Génère les prédictions IDNet et les pousse sur S3"
    )
    parser.add_argument(
        "--csv-path",
        type=str,
        default="data/processed/val.csv",
        help="Chemin vers le CSV manifest (colonnes: image_path, label)",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="predictions/test_run",
        help="Préfixe S3 de destination",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Nombre maximum d'images à traiter (None = tout)",
    )
    args = parser.parse_args()

    n = run(
        csv_path=args.csv_path,
        prefix=args.prefix,
        max_samples=args.max_samples,
    )
    print(f"\n{n} prédictions uploadées sur s3://{S3_BUCKET}/{args.prefix}/")
