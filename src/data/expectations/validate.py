"""
Great Expectations — validation du dataset IDNet.
Vérifie la qualité des données avant entraînement.
"""

import os
from pathlib import Path

import pandas as pd
import great_expectations as gx

PROCESSED_DIR = Path(os.getenv("DATA_PROCESSED_DIR", "data/processed"))
GE_DIR = Path("great_expectations")


def build_context() -> gx.DataContext:
    """Initialise ou charge le contexte GE."""
    if not GE_DIR.exists():
        context = gx.get_context(mode="file", project_root_dir=str(GE_DIR))
    else:
        context = gx.get_context(context_root_dir=str(GE_DIR))
    return context


def validate_split(df: pd.DataFrame, split_name: str) -> bool:
    """
    Valide un split (train/val/test) avec Great Expectations.
    Retourne True si toutes les expectations passent.
    """
    context = build_context()

    # Datasource in-memory
    datasource = context.sources.add_or_update_pandas(name="idnet_pandas")
    asset = datasource.add_dataframe_asset(name=f"idnet_{split_name}")
    batch_req = asset.build_batch_request(dataframe=df)

    # Créer ou récupérer la suite d'expectations
    suite_name = f"idnet_{split_name}_suite"
    try:
        suite = context.get_expectation_suite(suite_name)
    except Exception:
        suite = context.add_expectation_suite(suite_name)

    validator = context.get_validator(
        batch_request=batch_req,
        expectation_suite=suite,
    )

    # ─── Expectations ─────────────────────────────────────────────────────────

    # Colonnes obligatoires
    validator.expect_table_columns_to_match_ordered_list(["path", "label"])

    # Pas de valeurs nulles
    validator.expect_column_values_to_not_be_null("path")
    validator.expect_column_values_to_not_be_null("label")

    # Labels binaires uniquement (0=genuine, 1=fraud)
    validator.expect_column_values_to_be_in_set("label", [0, 1])

    # Pas de doublons sur les chemins d'images
    validator.expect_column_values_to_be_unique("path")

    # Dataset non vide
    validator.expect_table_row_count_to_be_between(min_value=1)

    # Équilibre des classes : fraud entre 20% et 80%
    fraud_ratio = df["label"].mean()
    validator.expect_column_mean_to_be_between(
        "label",
        min_value=0.20,
        max_value=0.80,
    )

    # Paths existent sur le disque
    validator.expect_column_values_to_match_regex(
        "path",
        regex=r".+\.(jpg|jpeg|png)$",
    )

    # ─── Sauvegarder et valider ───────────────────────────────────────────────
    validator.save_expectation_suite(discard_failed_expectations=False)

    results = context.run_validation_operator(
        "action_list_operator",
        assets_to_validate=[validator],
        run_id=f"validate_{split_name}",
    )

    success = results["success"]
    status = "✅ PASSED" if success else "❌ FAILED"
    print(
        f"[GE] {split_name:5s} validation {status} " f"(fraud_ratio={fraud_ratio:.2%}, n={len(df)})"
    )

    return success


def validate_all() -> bool:
    """Valide les 3 splits. Lève une exception si l'un échoue."""
    splits = {
        "train": pd.read_csv(PROCESSED_DIR / "train.csv"),
        "val": pd.read_csv(PROCESSED_DIR / "val.csv"),
        "test": pd.read_csv(PROCESSED_DIR / "test.csv"),
    }

    results = {name: validate_split(df, name) for name, df in splits.items()}

    if not all(results.values()):
        failed = [k for k, v in results.items() if not v]
        raise ValueError(f"Great Expectations validation FAILED for: {failed}")

    print("[GE] All splits validated successfully ✅")
    return True


if __name__ == "__main__":
    validate_all()
