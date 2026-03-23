import os
from pathlib import Path

import pandas as pd

import great_expectations as gx

PROCESSED_DIR = Path(os.getenv("DATA_PROCESSED_DIR", "data/processed"))
GE_DIR = Path("great_expectations")


def build_context():
    """Initialise le contexte GX moderne."""
    # GX 1.x gère seul la création/chargement
    return gx.get_context(project_root_dir=str(GE_DIR))


def validate_split(df: pd.DataFrame, split_name: str) -> bool:
    context = build_context()

    # 1. Datasource & Asset
    ds_name = "idnet_pandas"
    # On récupère ou on crée la source
    try:
        datasource = context.data_sources.get(ds_name)
    except KeyError:
        datasource = context.data_sources.add_pandas(name=ds_name)

    # On récupère ou on crée l'asset
    asset_name = f"idnet_{split_name}"
    try:
        asset = datasource.get_asset(asset_name)
    except (KeyError, LookupError, AttributeError):
        asset = datasource.add_dataframe_asset(name=asset_name)

    batch_req = asset.build_batch_request(options={"dataframe": df})

    # 2. Suite d'expectations
    suite_name = f"idnet_{split_name}_suite"

    # On définit la suite
    suite_obj = gx.ExpectationSuite(name=suite_name)

    # On utilise add_or_update au lieu de add() ou get()
    suite = context.suites.add_or_update(suite_obj)

    validator = context.get_validator(
        batch_request=batch_req,
        expectation_suite=suite,
    )
    # --- Expectations ---
    validator.expect_table_columns_to_match_ordered_list(["path", "label"])
    validator.expect_column_values_to_not_be_null("path")
    validator.expect_column_values_to_not_be_null("label")
    validator.expect_column_values_to_be_in_set("label", [0, 1])
    validator.expect_column_values_to_be_unique("path")
    validator.expect_table_row_count_to_be_between(min_value=1)

    fraud_ratio = df["label"].mean()
    # validator.expect_column_mean_to_be_between("label", min_value=0.20, max_value=0.80)
    validator.expect_column_values_to_be_in_set("label", [0, 1])

    validator.expect_column_values_to_match_regex("path", regex=r".+\.(?:jpg|jpeg|png)$")

    # 4. Sauvegarde et Validation
    # validator.save_expectation_suite()

    results = validator.validate()

    success = bool(results.success)
    status = "✅ PASSED" if success else "❌ FAILED"
    print(f"[GE] {split_name:5s} validation {status} (fraud_ratio={fraud_ratio:.2%}, n={len(df)})")

    return success


def validate_all() -> bool:
    """Valide les 3 splits."""
    for split in ["train", "val", "test"]:
        file_path = PROCESSED_DIR / f"{split}.csv"
        if not file_path.exists():
            print(f"⚠️ Skip {split}: file not found")
            continue

        df = pd.read_csv(file_path)
        if not validate_split(df, split):
            raise ValueError(f"Great Expectations validation FAILED for: {split}")

    print("[GE] All splits validated successfully ✅")
    return True


if __name__ == "__main__":
    validate_all()
