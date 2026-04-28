import os
from pathlib import Path

import mlflow.pytorch
import pandas as pd
import torch
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient
from torch.utils.data import DataLoader

from src.data.dataset import VAL_TF, IDNetDataset
from src.models.config import BATCH_SIZE, DEVICE

load_dotenv()

PROCESSED_DIR = Path(os.getenv("DATA_PROCESSED_DIR", "data/processed"))
MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "IDNet-Fraud-Detector")

# ─────────────────────────────────────────────
# Load model depuis MLflow (@champion)
# ─────────────────────────────────────────────

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))  # type: ignore

print("🏆 Loading champion model from MLflow...")
model = mlflow.pytorch.load_model(f"models:/{MODEL_NAME}@champion")
model.to(DEVICE)
model.eval()

# ─────────────────────────────────────────────
# Load deployment threshold
# ─────────────────────────────────────────────

client = MlflowClient()
mv = client.get_model_version_by_alias(MODEL_NAME, "champion")

threshold = float(mv.tags.get("deployment_threshold", 0.25))
print(f"🎯 Threshold utilisé: {threshold}")

# ─────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────

csv_path = PROCESSED_DIR / "train.csv"
df = pd.read_csv(csv_path)

loader = DataLoader(
    IDNetDataset(csv_path, VAL_TF),
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)

# ─────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────

all_scores, all_preds = [], []

with torch.no_grad():
    for images, _ in loader:
        probs = torch.softmax(model(images.to(DEVICE)), dim=1)[:, 1]
        scores = probs.cpu().numpy()

        all_scores.extend(scores.tolist())
        all_preds.extend((scores >= threshold).astype(int).tolist())

df["score"] = all_scores
df["prediction"] = all_preds

# ─────────────────────────────────────────────
# Save
# ─────────────────────────────────────────────

output = PROCESSED_DIR / "train_with_preds.csv"
df.to_csv(output, index=False)

print(f"✅ Généré : {output} ({len(df)} lignes)")
