"""
Pipeline data complet — une seule commande pour tout lancer.

Usage:
    python -m src.data.pipeline              # download + preprocess + validate
    python -m src.data.pipeline --no-download  # preprocess + validate seulement
    python -m src.data.pipeline --no-validate  # download + preprocess seulement
"""

import argparse

from src.data.download import download_idnet
from src.data.preprocess import preprocess
from src.data.expectations.validate import validate_all


def run(download: bool = True, validate: bool = True) -> None:
    if download:
        print("\n── Étape 1/3 : Téléchargement IDNet ──────────────────────────")
        download_idnet()

    print("\n── Étape 2/3 : Prétraitement & split ─────────────────────────")
    preprocess()

    if validate:
        print("\n── Étape 3/3 : Validation Great Expectations ─────────────────")
        validate_all()

    print("\n✅ Pipeline data terminé.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline data IDNet")
    parser.add_argument("--no-download", action="store_true", help="Skip download")
    parser.add_argument("--no-validate", action="store_true", help="Skip GE validation")
    args = parser.parse_args()

    run(
        download=not args.no_download,
        validate=not args.no_validate,
    )
