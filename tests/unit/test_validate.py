import pandas as pd
import pytest

from src.data.expectations.validate import validate_split

# ---------------------------------------------------------------------------
# Valid manifest → validation must pass
# ---------------------------------------------------------------------------


class TestValidatePassesOnGoodManifest:
    def test_valid_manifest_passes(self):
        # On crée un DF valide (ratio fraude 50% pour passer le check 20-80%)
        df = pd.DataFrame({"path": ["img1.png", "img2.jpg"], "label": [0, 1]})

        result = validate_split(df, "test_pass")
        assert result is True, "Un dataset valide devrait passer"


# ---------------------------------------------------------------------------
# Invalid manifests → validation must fail
# ---------------------------------------------------------------------------


class TestValidateFailsOnBadManifest:
    def test_missing_label_column_fails(self):
        df = pd.DataFrame({"path": ["img1.png"]})

        # On attrape une erreur spécifique (ValueError ou KeyError selon l'implémentation)
        with pytest.raises((ValueError, KeyError, Exception)):
            validate_split(df, "test_missing_col")

    def test_invalid_label_value_fails(self):
        """Label '99' est hors du domaine {0, 1}."""
        df = pd.DataFrame({"path": ["img1.png"], "label": [99]})
        result = validate_split(df, "test_bad_label")
        assert result is False, "Le label 99 devrait faire échouer la validation"

    def test_empty_df_fails(self):
        df = pd.DataFrame(columns=["path", "label"])
        result = validate_split(df, "test_empty")
        assert result is False, "Un dataset vide devrait échouer"


# ---------------------------------------------------------------------------
# Fraud ratio expectation
# ---------------------------------------------------------------------------


class TestFraudRatioExpectation:
    def test_all_genuine_fails_ratio_check(self):
        """100 % genuine (0 % fraud) doit échouer car min=20% dans le script."""
        df = pd.DataFrame({"path": [f"img_{i}.png" for i in range(10)], "label": [0] * 10})

        result = validate_split(df, "test_ratio")
        assert result is False, "0% de fraude devrait échouer (attendu > 20%)"
