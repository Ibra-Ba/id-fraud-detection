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
) -> None:
    """
    Pipeline data IDNet : Téléchargement -> Preprocess -> Validation.
    Supporte l'injection de dossiers pour les tests unitaires via arguments.
    """
    # Force la conversion en Path si des strings sont passées
    raw_dir = Path(raw_dir)
    processed_dir = Path(processed_dir)

    if not skip_download:
        print("\n── Étape 1/3 : Téléchargement IDNet ──────────────────────────")
        # Si download_idnet ne supporte pas d'argument,
        # elle utilisera sa config par défaut.
        download_idnet()

    print("\n── Étape 2/3 : Prétraitement & split ─────────────────────────")
    # On passe explicitement les dossiers
    preprocess(raw_dir=raw_dir, processed_dir=processed_dir)

    if validate:
        print("\n── Étape 3/3 : Validation Great Expectations ─────────────────")
        # On définit la variable d'environnement pour que validate_all()
        # sache quel dossier processed valider (utile pour l'isolation des tests)
        os.environ["DATA_PROCESSED_DIR"] = str(processed_dir)
        validate_all()

    print("\n✅ Pipeline data terminé.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline data IDNet")
    parser.add_argument("--no-download", action="store_true", help="Skip download")
    parser.add_argument("--no-validate", action="store_true", help="Skip GE validation")
    # Optionnel : permettre de passer les chemins via CLI
    parser.add_argument("--raw-dir", type=str, default="data/raw")
    parser.add_argument("--processed-dir", type=str, default="data/processed")

    args = parser.parse_args()

    run_pipeline(
        raw_dir=Path(args.raw_dir),
        processed_dir=Path(args.processed_dir),
        skip_download=args.no_download,
        validate=not args.no_validate,
    )
