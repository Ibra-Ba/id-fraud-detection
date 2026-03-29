import argparse
import os
from pathlib import Path

from src.data.download import download_idnet
from src.data.expectations.validate import validate_all
from src.data.preprocess import preprocess


def run_pipeline(
    raw_dir: Path = Path("data/raw"),
    processed_dir: Path = Path("data/processed"),
    skip_download: bool = False,
    validate: bool = True,
    upload_s3: bool = False,
    manifests_only: bool = False,
) -> None:
    """
    Pipeline data IDNet : Téléchargement -> Preprocess -> Validation -> Upload S3.
    Supporte l'injection de dossiers pour les tests unitaires via arguments.
    """
    raw_dir = Path(raw_dir)
    processed_dir = Path(processed_dir)

    if not skip_download:
        print("\n── Étape 1/4 : Téléchargement IDNet ──────────────────────────")
        download_idnet()

    print("\n── Étape 2/4 : Prétraitement & split ─────────────────────────")
    preprocess()

    if validate:
        print("\n── Étape 3/4 : Validation Great Expectations ─────────────────")
        os.environ["DATA_PROCESSED_DIR"] = str(processed_dir)
        validate_all()

    if upload_s3:
        print("\n── Étape 4/4 : Upload S3 ──────────────────────────────────────")
        from src.data.upload_to_s3 import upload_all

        upload_all(manifests_only=manifests_only)

    print("\n✅ Pipeline data terminé.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline data IDNet")
    parser.add_argument("--no-download", action="store_true", help="Skip download")
    parser.add_argument("--no-validate", action="store_true", help="Skip GE validation")
    parser.add_argument(
        "--upload-s3",
        dest="upload_s3",
        action="store_true",
        help="Upload données vers S3 après validation",
    )
    parser.add_argument(
        "--manifests-only",
        dest="manifests_only",
        action="store_true",
        help="Upload CSV seulement (pas les images)",
    )
    parser.add_argument("--raw-dir", type=str, default="data/raw")
    parser.add_argument("--processed-dir", type=str, default="data/processed")

    args = parser.parse_args()

    run_pipeline(
        raw_dir=Path(args.raw_dir),
        processed_dir=Path(args.processed_dir),
        skip_download=args.no_download,
        validate=not args.no_validate,
        upload_s3=args.upload_s3,
        manifests_only=args.manifests_only,
    )
