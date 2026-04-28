"""
Examples Usage:
 python -m src.monitoring.generate_predictions \
    --csv data/processed/test.csv \
        --prefix predictions/test_run \
            --max-samples 500
"""

import json
import logging
import os
import time
from datetime import datetime

import boto3
import mlflow
import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient
from PIL import Image
from torch.utils.data import DataLoader

from src.data.dataset import VAL_TF, IDNetDataset
from src.models.config import BATCH_SIZE, DEVICE

# Logging config

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

load_dotenv()

S3_BUCKET = os.getenv("S3_BUCKET")
s3 = boto3.client("s3")


def get_champion_threshold() -> float:
    uri = os.getenv("MLFLOW_TRACKING_URI")
    if not uri:
        logger.warning("MLFLOW_TRACKING_URI non défini, seuil par défaut 0.25")
        return 0.25

    mlflow.set_tracking_uri(uri)
    client = MlflowClient()
    model_name = os.getenv("MLFLOW_MODEL_NAME", "IDNet-Fraud-Detector")

    try:
        mv = client.get_model_version_by_alias(model_name, "champion")

        threshold = float(mv.tags.get("deployment_threshold", 0.25))

        logger.info(f"🎯 Threshold (deployment) récupéré " f"(V{mv.version}) : {threshold}")
        return threshold

    except Exception as e:
        logger.error(f"❌ Impossible de récupérer le seuil : {e}")
        return 0.25


# Load model


def load_model():
    """Charge le modèle champion depuis MLflow avec vérification stricte de l'URI."""
    import mlflow.pytorch

    uri = os.getenv("MLFLOW_TRACKING_URI")
    model_name = os.getenv("MLFLOW_MODEL_NAME", "IDNet-Fraud-Detector")
    if not uri:
        logger.error("❌ MLFLOW_TRACKING_URI n'est pas définie dans le .env. Arrêt du script.")
        exit(1)

    try:
        mlflow.set_tracking_uri(uri)
        logger.info(f"🏆 Chargement du modèle champion '{model_name}' depuis {uri}")

        model = mlflow.pytorch.load_model(f"models:/{model_name}@champion")

        model.eval()
        model.to(DEVICE)
        logger.info("✅ Modèle chargé avec succès")
        return model

    except Exception as e:
        logger.error(f"❌ Impossible de charger le modèle champion : {e}")
        exit(1)
    return model


# Upload to S3


def upload_to_s3(records, prefix, max_retries=3):
    # logger.info(f"Uploading {len(records)} predictions to S3 ({prefix})...")
    total = len(records)
    logger.info(f"📤 Starting S3 upload: {total} files to bucket '{S3_BUCKET}'")
    start_upload = time.time()
    success_count = 0
    error_count = 0

    for i, r in enumerate(records):
        key = f"{prefix}/{r['id']}.json"

        # Log de progression tous les 50 fichiers
        if i % 50 == 0 and i > 0:
            elapsed = time.time() - start_upload
            speed = i / elapsed
            logger.info(f"⏳ Progress: {i}/{total} uploaded... ({speed:.1f} files/s)")
        for attempt in range(max_retries):
            try:
                s3.put_object(
                    Bucket=S3_BUCKET,
                    Key=key,
                    Body=json.dumps(r),
                    ContentType="application/json",
                )
                success_count += 1
                break  # succès

            except Exception as e:
                logger.warning(
                    f"S3 upload failed (attempt {attempt+1}/{max_retries}) " f"for {key}: {e}"
                )
                time.sleep(1)

        else:
            logger.error(f"FAILED after {max_retries} attempts: {key}")

    total_time = time.time() - start_upload
    final_speed = total / total_time if total_time > 0 else 0

    logger.info("── S3 Upload Report ──────────────────────────")
    logger.info(f"✅ Successfully uploaded : {success_count}")
    logger.info(f"❌ Failed                : {error_count}")
    logger.info(f"⏱️  Total time           : {total_time:.2f}s")
    logger.info(f"🚀 Average speed        : {final_speed:.1f} files/s")
    logger.info("──────────────────────────────────────────────")


# Main batch job


def run(csv_path: str, prefix: str, max_samples: int = 500):
    start_time = time.time()

    logger.info("── Batch prediction job started ─────────────")
    logger.info(f"Input CSV: {csv_path}")

    df = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(df)} samples")

    if len(df) > max_samples:
        df = df.sample(max_samples, random_state=42)
        logger.info(f"Sampling applied → {len(df)} samples")

    model = load_model()

    # Récupérer threshold dynmique
    threshold = get_champion_threshold()
    logger.info(f"Threshold lu depuis MLflow : {threshold}")

    dataset = IDNetDataset(csv_path, VAL_TF)
    loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True
    )

    records = []
    total_batches = len(loader)
    fraud_detected = 0

    logger.info(f"Starting inference ({total_batches} batches)...")
    # Charge le CSV pour accéder aux chemins des images
    df_paths = pd.read_csv(csv_path).reset_index(drop=True)
    idx = 0
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(loader):
            batch_start = time.time()
            probs = torch.softmax(model(images.to(DEVICE)), dim=1)[:, 1]

            for i in range(len(probs)):
                # Charge l'image pour les features pixel
                try:
                    img = Image.open(df_paths.iloc[idx]["path"]).convert("RGB")
                    img_arr = np.array(img, dtype=np.float32) / 255.0

                except Exception as e:
                    img_arr = np.zeros((224, 224, 3), dtype=np.float32)
                    logger.warning(f"Failed to process image at index {idx}: {e}")

                score = float(probs[i])
                is_fraud = int(score >= threshold)
                if is_fraud:
                    fraud_detected += 1

                records.append(
                    {
                        "id": f"{idx}",
                        "timestamp": datetime.utcnow().isoformat(),
                        "label": int(labels[i]),
                        "prediction": int(probs[i] >= threshold),
                        "score": float(probs[i]),
                        # Features pixels
                        "r_mean": float(img_arr[:, :, 0].mean()),
                        "g_mean": float(img_arr[:, :, 1].mean()),
                        "b_mean": float(img_arr[:, :, 2].mean()),
                        # Std
                        "r_std": float(img_arr[:, :, 0].std()),
                        "g_std": float(img_arr[:, :, 1].std()),
                        "b_std": float(img_arr[:, :, 2].std()),
                        "luminosity_mean": float(img_arr.mean()),
                        "luminosity_std": float(img_arr.std()),
                        "contrast": float(img_arr.max() - img_arr.min()),
                        "colorfulness": float(
                            np.std(img_arr[:, :, 0] - img_arr[:, :, 1])
                            + np.std(
                                0.5 * (img_arr[:, :, 0] + img_arr[:, :, 1]) - img_arr[:, :, 2]
                            ),
                        ),
                    }
                )
                idx += 1

            # Log de progression par batch
            if batch_idx % 5 == 0 or batch_idx == total_batches - 1:
                batch_time = time.time() - batch_start
                logger.info(
                    f"📦 Batch {batch_idx+1}/{total_batches} | "
                    f"Speed: {batch_time:.2f}s/batch | "
                    f"Current Fraud Rate: {(fraud_detected/idx):.2%}"
                )
    logger.info(f"✅ Inference completed → {len(records)} predictions generated")
    logger.info(
        f"📊 Global Summary: {fraud_detected} frauds detected (Rate: {fraud_detected/len(records):.2%})"
    )

    upload_to_s3(records, prefix)

    elapsed = time.time() - start_time
    logger.info(f"── Job completed in {elapsed:.2f}s ─────────────")


# CLI

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--prefix", type=str, required=True)
    parser.add_argument("--max-samples", type=int, default=500)

    args = parser.parse_args()

    run(args.csv, args.prefix, args.max_samples)
