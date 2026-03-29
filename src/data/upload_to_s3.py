"""
Upload des données preprocessées et des images raw vers S3.
Fait partie du pipeline data — déclenché après validate.

Usage:
    python -m src.data.upload_to_s3                    # upload tout
    python -m src.data.upload_to_s3 --manifests-only   # CSV seulement
"""

import argparse
import logging
import os
from pathlib import Path

import boto3
from botocore.exceptions import BotoCoreError, ClientError
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

S3_BUCKET = os.getenv("S3_BUCKET")
RAW_DIR = Path(os.getenv("DATA_RAW_DIR", "data/raw"))
PROCESSED_DIR = Path(os.getenv("DATA_PROCESSED_DIR", "data/processed"))


def _upload_dir(s3_client, local_dir: Path, s3_prefix: str, extensions: set[str]) -> int:
    """
    Upload récursif d'un dossier local vers S3.
    Retourne le nombre de fichiers uploadés.
    """
    count = 0
    for file_path in local_dir.rglob("*"):
        if not file_path.is_file():
            continue
        if file_path.suffix.lower() not in extensions:
            continue

        # Clé S3 relative au dossier local
        relative = file_path.relative_to(local_dir)
        s3_key = f"{s3_prefix}/{relative}".replace("\\", "/")

        try:
            s3_client.upload_file(str(file_path), S3_BUCKET, s3_key)
            logger.info(f"  ✅ {s3_key}")
            count += 1
        except (BotoCoreError, ClientError) as e:
            logger.error(f"  ❌ Échec upload {s3_key} : {e}")
            raise

    return count


def upload_manifests(s3_client) -> int:
    """Upload les CSV manifests (train/val/test) vers S3."""
    logger.info(f"\n── Upload manifests CSV → s3://{S3_BUCKET}/data/processed/")
    count = _upload_dir(s3_client, PROCESSED_DIR, "data/processed", {".csv"})
    logger.info(f"   {count} fichier(s) uploadé(s)")
    return count


def upload_images(s3_client) -> int:
    """Upload les images raw vers S3."""
    logger.info(f"\n── Upload images raw → s3://{S3_BUCKET}/data/raw/")
    count = _upload_dir(s3_client, RAW_DIR, "data/raw", {".jpg", ".jpeg", ".png"})
    logger.info(f"   {count} fichier(s) uploadé(s)")
    return count


def upload_all(manifests_only: bool = False) -> dict:
    """Point d'entrée principal."""
    if not S3_BUCKET:
        raise RuntimeError("S3_BUCKET non défini dans les variables d'environnement")

    s3_client = boto3.client(
        "s3",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=os.getenv("AWS_DEFAULT_REGION", "eu-west-3"),
    )

    results = {}
    results["manifests"] = upload_manifests(s3_client)

    if not manifests_only:
        results["images"] = upload_images(s3_client)

    logger.info("\n✅ Upload S3 terminé.")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload données IDNet vers S3")
    parser.add_argument(
        "--manifests-only",
        dest="manifests_only",
        action="store_true",
        help="Upload uniquement les CSV manifests (pas les images)",
    )
    args = parser.parse_args()
    upload_all(manifests_only=args.manifests_only)
