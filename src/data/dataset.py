"""
Dataset PyTorch et transforms pour IDNet.
Partagés entre train.py et evaluate.py.
"""

import albumentations as A
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import Dataset

from src.models.config import IMG_SIZE

# ─── Transforms ───────────────────────────────────────────────────────────────
TRAIN_TF = A.Compose(
    [
        A.Resize(*IMG_SIZE),
        A.HorizontalFlip(p=0.3),
        A.RandomBrightnessContrast(p=0.4),
        A.HueSaturationValue(p=0.3),
        A.Perspective(p=0.2),
        A.GaussNoise(p=0.2),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)

VAL_TF = A.Compose(
    [
        A.Resize(*IMG_SIZE),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)


# ─── Dataset ──────────────────────────────────────────────────────────────────
class IDNetDataset(Dataset):
    def __init__(self, csv_path, transform: A.Compose):
        self.df = pd.read_csv(csv_path)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        row = self.df.iloc[idx]
        image = np.array(Image.open(row["path"]).convert("RGB"))
        image = self.transform(image=image)["image"]
        return image, int(row["label"])
