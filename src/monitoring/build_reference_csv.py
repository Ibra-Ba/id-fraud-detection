"""
Construction du CSV de référence — IDNet Fraud Detector Bac+5

Reconstruit train_with_preds.csv en agrégeant les fichiers JSON
stockés sous s3://{bucket}/predictions/train_ref/*.json
(produits par generate_predictions.py).

Le CSV résultant est :
  1. Sauvegardé localement dans data/processed/train_with_preds.csv
  2. Uploadé sur S3 pour persistance
  3. Prêt à être loggué comme artifact MLflow par run_monitoring_pipeline.py

Colonnes produites (attendues par monitor_pro.py) :
    id, label, score, prediction, timestamp

Usage:
    python -m src.monitoring.build_reference_csv
    python -m src.monitoring.build_reference_csv \\
        --s3-prefix predictions/train_ref \\
        --output data/processed/train_with_preds.csv
"""

import argparse
import json
import logging
import os
from pathlib import Path

import boto3
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Constantes ────────────────────────────────────────────────────────────────
S3_BUCKET = os.getenv("S3_BUCKET")
S3_PREFIX_DEFAULT = "predictions/train_ref"
OUTPUT_PATH_DEFAULT = "data/processed/train_with_preds.csv"
S3_OUTPUT_KEY = "data/processed/train_with_preds.csv"


# ── Chargement S3 ─────────────────────────────────────────────────────────────


def load_records_from_s3(prefix: str) -> list[dict]:
    """
    Charge tous les fichiers JSON depuis s3://{bucket}/{prefix}/*.json.
    Cohérent avec load_s3_predictions() dans monitor_pro.py.
    """
    if not S3_BUCKET:
        raise RuntimeError("S3_BUCKET non défini dans l'environnement")

    s3 = boto3.client("s3")
    records = []

    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if not key.endswith(".json"):
                continue

            response = s3.get_object(Bucket=S3_BUCKET, Key=key)
            content = json.loads(response["Body"].read())
            records.append(content)

    if not records:
        raise ValueError(
            f"Aucun fichier JSON trouvé dans s3://{S3_BUCKET}/{prefix}/. "
            f"Lancez d'abord generate_predictions.py avec --prefix {prefix}"
        )

    logger.info(f"{len(records)} records chargés depuis s3://{S3_BUCKET}/{prefix}/")
    return records


# ── Construction du DataFrame ─────────────────────────────────────────────────


def build_dataframe(records: list[dict]) -> pd.DataFrame:
    """
    Construit et valide le DataFrame de référence.
    Colonnes attendues par monitor_pro.py : label, score, prediction.
    """
    df = pd.DataFrame(records)

    if df.empty:
        return df

    # Typage explicite (cohérent avec load_data() dans monitor_pro.py)
    if "score" in df.columns:
        df["score"] = df["score"].astype(float)
    if "prediction" in df.columns:
        df["prediction"] = df["prediction"].astype(int)

    if "label" in df.columns:
        # Filtre les lignes sans label (label=None = inférence sans vérité terrain)
        n_before = len(df)
        df = df[df["label"].notna()].copy()
        df["label"] = df["label"].astype(int)
        n_dropped = n_before - len(df)
        if n_dropped > 0:
            logger.warning(f"{n_dropped} lignes sans label supprimées")

    # Supprime les colonnes inutiles pour le monitoring
    df.drop(columns=["path", "image_path"], errors="ignore", inplace=True)

    # Supprime les doublons sur id si présent
    if "id" in df.columns:
        n_before = len(df)
        df = df.drop_duplicates(subset=["id"])
        n_dup = n_before - len(df)
        if n_dup > 0:
            logger.warning(f"{n_dup} doublons supprimés sur la colonne 'id'")

    # Ordre des colonnes : id en premier si présent, puis le reste
    priority_cols = [
        c for c in ["id", "label", "score", "prediction", "timestamp"] if c in df.columns
    ]
    other_cols = [c for c in df.columns if c not in priority_cols]
    df = df[priority_cols + other_cols]

    logger.info(f"DataFrame construit : {len(df)} lignes, colonnes={list(df.columns)}")

    if "label" in df.columns:
        fraud_rate = df["label"].mean()
        logger.info(f"Taux de fraude dans la référence : {fraud_rate:.2%}")

    return df


# ── Sauvegarde ────────────────────────────────────────────────────────────────


def save_local(df: pd.DataFrame, output_path: str) -> Path:
    """Sauvegarde le CSV localement."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info(f"CSV sauvegardé localement : {path} ({len(df)} lignes)")
    return path


def upload_to_s3(local_path: Path, s3_key: str) -> None:
    """Upload le CSV sur S3 pour persistance."""
    if not S3_BUCKET:
        raise RuntimeError("S3_BUCKET non défini")

    s3 = boto3.client("s3")
    s3.upload_file(str(local_path), S3_BUCKET, s3_key)
    logger.info(f"CSV uploadé sur s3://{S3_BUCKET}/{s3_key}")


# ── Point d'entrée principal ──────────────────────────────────────────────────


def main(
    s3_prefix: str = S3_PREFIX_DEFAULT,
    output_path: str = OUTPUT_PATH_DEFAULT,
    upload: bool = True,
) -> Path:
    """
    Pipeline complet : S3 JSONs → DataFrame → CSV local → S3.

    Args:
        s3_prefix: préfixe S3 source (ex: predictions/train_ref)
        output_path: chemin local de sortie
        upload: si True, uploade aussi le CSV sur S3

    Returns:
        Path vers le CSV local produit
    """
    logger.info("── build_reference_csv démarré ─────────────────────────────")

    # 1. Charge les JSONs depuis S3
    records = load_records_from_s3(s3_prefix)

    # 2. Construit le DataFrame
    df = build_dataframe(records)

    # 3. Sauvegarde locale
    local_path = save_local(df, output_path)

    # 4. Upload S3
    if upload:
        upload_to_s3(local_path, S3_OUTPUT_KEY)

    logger.info("── build_reference_csv terminé ─────────────────────────────")
    return local_path


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Reconstruit train_with_preds.csv depuis les prédictions S3"
    )
    parser.add_argument(
        "--s3-prefix",
        type=str,
        default=S3_PREFIX_DEFAULT,
        help=f"Préfixe S3 source (défaut: {S3_PREFIX_DEFAULT})",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=OUTPUT_PATH_DEFAULT,
        help=f"Chemin local de sortie (défaut: {OUTPUT_PATH_DEFAULT})",
    )
    parser.add_argument(
        "--no-upload",
        action="store_true",
        help="Ne pas uploader le CSV sur S3 (mode local uniquement)",
    )
    args = parser.parse_args()

    path = main(
        s3_prefix=args.s3_prefix,
        output_path=args.output,
        upload=not args.no_upload,
    )
    print(f"\nCSV de référence produit : {path}")
