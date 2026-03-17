"""
Preprocess IDNet dataset:
- Split into train / val / test (70 / 15 / 15)
- Apply Albumentations augmentation on train set
- Save processed splits as CSV manifests
"""

import os
from pathlib import Path

import albumentations as A
import pandas as pd
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split

RAW_DIR = Path(os.getenv("DATA_RAW_DIR", "data/raw"))
PROCESSED_DIR = Path(os.getenv("DATA_PROCESSED_DIR", "data/processed"))
IMG_SIZE = (224, 224)
RANDOM_SEED = 42

# ─── Albumentations pipelines ────────────────────────────────────────────────

TRAIN_TRANSFORMS = A.Compose(
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

VAL_TEST_TRANSFORMS = A.Compose(
    [
        A.Resize(*IMG_SIZE),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)


def _collect_images(raw_dir: Path) -> pd.DataFrame:
    """
    Walk raw_dir and collect all images with their label.
    IDNet structure: raw/<Country>/<split>/<label>/<image>.jpg
    label: 'genuine' → 0, everything else → 1 (fraud)
    """
    records = []
    for img_path in raw_dir.rglob("*.jpg"):
        parts = img_path.parts
        # IDNet labels folder is 2 levels up from image
        label_folder = parts[-2].lower()
        label = 0 if label_folder == "genuine" else 1
        records.append({"path": str(img_path), "label": label})

    if not records:
        raise FileNotFoundError(f"No images found in {raw_dir}. Run download.py first.")

    df = pd.DataFrame(records)
    print(f"[INFO] Found {len(df)} images — {df.label.value_counts().to_dict()}")
    return df


def split_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified 70/15/15 split."""
    train, temp = train_test_split(
        df, test_size=0.30, stratify=df["label"], random_state=RANDOM_SEED
    )
    val, test = train_test_split(
        temp, test_size=0.50, stratify=temp["label"], random_state=RANDOM_SEED
    )
    return train, val, test


def save_manifests(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> None:
    """Save CSV manifests for each split."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    train.to_csv(PROCESSED_DIR / "train.csv", index=False)
    val.to_csv(PROCESSED_DIR / "val.csv", index=False)
    test.to_csv(PROCESSED_DIR / "test.csv", index=False)
    print(f"[OK] Manifests saved → {PROCESSED_DIR}")
    print(f"     train={len(train)} | val={len(val)} | test={len(test)}")


def preprocess() -> None:
    df = _collect_images(RAW_DIR)
    train, val, test = split_dataset(df)
    save_manifests(train, val, test)


if __name__ == "__main__":
    preprocess()


def preprocess_with_validation() -> None:
    """
    Pipeline complet :
    1. Collecte les images IDNet
    2. Split stratifié 70/15/15
    3. Sauvegarde les manifests CSV
    4. Validation Great Expectations
    """
    df = _collect_images(RAW_DIR)
    train, val, test = split_dataset(df)
    save_manifests(train, val, test)

    from src.data.expectations.validate import validate_all

    validate_all()
