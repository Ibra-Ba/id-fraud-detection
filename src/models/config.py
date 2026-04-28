"""
Constantes partagées entre train.py, evaluate.py et serving.
"""

import os
from pathlib import Path

# ─── Chemins ──────────────────────────────────────────────────────────────────
PROCESSED_DIR = Path(os.getenv("DATA_PROCESSED_DIR", "data/processed"))

# ─── Device ───────────────────────────────────────────────────────────────────
DEVICE = "cpu"  # pas de GPU requis

# ─── Modèle ───────────────────────────────────────────────────────────────────
NUM_CLASSES = 2  # binaire : genuine (0) vs fraud (1)
IMAGE_SIZE = 224  # côté carré attendu par EfficientNet-B0

# ─── DataLoader ───────────────────────────────────────────────────────────────
NUM_WORKERS = 0  # obligatoire sous WSL2 (pas de fork multiprocessing)

# ─── Hyperparamètres ──────────────────────────────────────────────────────────
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 16))
FREEZE_EPOCHS = int(os.getenv("FREEZE_EPOCHS", 3))
TOTAL_EPOCHS = int(os.getenv("TOTAL_EPOCHS", 25))
LR_HEAD = float(os.getenv("LR_HEAD", 1e-3))
LR_FINETUNE = float(os.getenv("LR_FINETUNE", 3e-5))
IMG_SIZE = (IMAGE_SIZE, IMAGE_SIZE)  # tuple pour Albumentations / torchvision

# ─── Quality gates ────────────────────────────────────────────────────────────
MIN_AUROC = float(os.getenv("MIN_AUROC", 0.90))
MIN_F1 = float(os.getenv("MIN_F1", 0.85))
