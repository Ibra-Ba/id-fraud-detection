"""
Preprocess IDNet ESP dataset:
- Structure : positive/ (genuine) + fraud*/ (fraud)
- Split stratifié 70/15/15
- Sauvegarde manifests CSV
"""

import os
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

RAW_DIR = Path(os.getenv("DATA_RAW_DIR", "data/raw"))
PROCESSED_DIR = Path(os.getenv("DATA_PROCESSED_DIR", "data/processed"))
RANDOM_SEED = 42

# Dossiers genuines et fraudes dans IDNet ESP
GENUINE_DIR = "positive"
FRAUD_DIRS = [
    "fraud1_copy_and_move",
    "fraud2_face_morphing",
    "fraud3_face_replacement",
    "fraud4_combined",
    "fraud5_inpaint_and_rewrite",
    "fraud6_crop_and_replace",
]


def _collect_images(raw_dir: Path) -> pd.DataFrame:
    """
    Collecte toutes les images IDNet ESP avec leurs labels.
    positive/  → label 0 (genuine)
    fraud*/    → label 1 (fraud)
    """
    records = []
    esp_dir = raw_dir / "ESP"

    if not esp_dir.exists():
        raise FileNotFoundError(
            f"Dossier ESP introuvable dans {raw_dir}. "
            "Lance d'abord : python -m src.data.download"
        )

    # ── Genuine ───────────────────────────────────────────────────────────────
    genuine_dir = esp_dir / GENUINE_DIR
    if genuine_dir.exists():
        genuine_imgs = (
            list(genuine_dir.rglob("*.jpg"))
            + list(genuine_dir.rglob("*.jpeg"))
            + list(genuine_dir.rglob("*.png"))
        )
        for img_path in genuine_imgs:
            records.append({"path": str(img_path), "label": 0})
        print(f"[INFO] Genuine (positive) : {len(genuine_imgs)} images")
    else:
        print(f"[WARN] Dossier genuine introuvable : {genuine_dir}")

    # ── Fraud ─────────────────────────────────────────────────────────────────
    fraud_count = 0
    for fraud_dir_name in FRAUD_DIRS:
        fraud_dir = esp_dir / fraud_dir_name
        if not fraud_dir.exists():
            print(f"[WARN] Dossier fraude introuvable : {fraud_dir}")
            continue
        fraud_imgs = (
            list(fraud_dir.rglob("*.jpg"))
            + list(fraud_dir.rglob("*.jpeg"))
            + list(fraud_dir.rglob("*.png"))
        )
        for img_path in fraud_imgs:
            records.append({"path": str(img_path), "label": 1})
        fraud_count += len(fraud_imgs)
        print(f"[INFO] {fraud_dir_name} : {len(fraud_imgs)} images")

    print(f"[INFO] Total fraud : {fraud_count} images")

    if not records:
        raise FileNotFoundError(f"Aucune image trouvée dans {esp_dir}")

    df = pd.DataFrame(records)
    print(f"\n[INFO] Total : {len(df)} images")
    print(f"       Genuine : {(df.label==0).sum()} | Fraud : {(df.label==1).sum()}")
    print(f"       Ratio fraud : {df.label.mean():.2%}")
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
    """Sauvegarde les CSV manifests pour chaque split."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    train.to_csv(PROCESSED_DIR / "train.csv", index=False)
    val.to_csv(PROCESSED_DIR / "val.csv", index=False)
    test.to_csv(PROCESSED_DIR / "test.csv", index=False)
    print(f"\n[OK] Manifests sauvegardés → {PROCESSED_DIR}")
    print(f"     train={len(train)} | val={len(val)} | test={len(test)}")


def preprocess() -> None:
    df = _collect_images(RAW_DIR)
    train, val, test = split_dataset(df)
    save_manifests(train, val, test)


if __name__ == "__main__":
    preprocess()
