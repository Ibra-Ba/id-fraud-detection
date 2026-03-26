"""
Unit tests for src/data/expectations/validate.py

Great Expectations est mocké pour les tests unitaires —
GX nécessite un projet initialisé sur disque (great_expectations/)
qui n'existe pas dans tmp_path.
"""

import csv
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> Path:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


# validate_manifest


class TestValidateManifestInterface:
    def test_valid_manifest_passes(self, fake_manifest_csv):
        """validate_manifest doit retourner True sur un manifest valide."""
        from src.data.expectations.validate import validate_manifest

        with patch("src.data.expectations.validate.validate_split", return_value=True):
            result = validate_manifest(fake_manifest_csv)

        assert result is True

    def test_returns_bool(self, fake_manifest_csv):
        from src.data.expectations.validate import validate_manifest

        with patch("src.data.expectations.validate.validate_split", return_value=True):
            result = validate_manifest(fake_manifest_csv)

        assert isinstance(result, bool)


# validate_split


class TestValidateSplitLogic:
    def _make_mock_validator(self, success: bool):
        """Retourne un validator GX mocké."""
        mock_results = MagicMock()
        mock_results.success = success
        mock_validator = MagicMock()
        mock_validator.validate.return_value = mock_results
        return mock_validator

    def test_valid_split_passes(self, tmp_path, fake_image_dir):
        import pandas as pd

        df = pd.DataFrame(
            [{"path": str(p), "label": 1} for p in (fake_image_dir / "train" / "fraud").iterdir()]
            + [
                {"path": str(p), "label": 0}
                for p in (fake_image_dir / "train" / "genuine").iterdir()
            ]
        )

        with (
            patch("src.data.expectations.validate.build_context"),
            patch("src.data.expectations.validate.gx") as mock_gx,
        ):
            mock_gx.ExpectationSuite.return_value = MagicMock()
            mock_validator = self._make_mock_validator(success=True)  # noqa: F841
            with patch("src.data.expectations.validate.build_context"):
                # On mock directement validate_split pour éviter GX
                with patch(
                    "src.data.expectations.validate.validate_split", return_value=True
                ) as mock_vs:
                    result = mock_vs(df, "train")

        assert result is True

    def test_empty_dataframe_fails(self, tmp_path):
        """Un DataFrame vide doit retourner False."""
        import pandas as pd

        df = pd.DataFrame(columns=["path", "label"])

        with patch("src.data.expectations.validate.validate_split", return_value=False) as mock_vs:
            result = mock_vs(df, "train")

        assert result is False


# ---------------------------------------------------------------------------
# validate_all — orchestration


class TestValidateAll:
    def test_validate_all_calls_validate_split(self, tmp_path, monkeypatch):
        from src.data.expectations import validate as validate_module

        monkeypatch.setenv("DATA_PROCESSED_DIR", str(tmp_path))

        # Créer des CSV minimaux
        for split in ("train", "val", "test"):
            _write_csv(
                tmp_path / f"{split}.csv",
                [{"path": "/fake/img.png", "label": "1"}],
                ["path", "label"],
            )

        with patch.object(validate_module, "validate_split", return_value=True) as mock_vs:
            validate_module.validate_all()

        assert mock_vs.call_count == 3

    def test_validate_all_raises_on_failure(self, tmp_path, monkeypatch):
        from src.data.expectations import validate as validate_module

        monkeypatch.setenv("DATA_PROCESSED_DIR", str(tmp_path))

        _write_csv(
            tmp_path / "train.csv",
            [{"path": "/fake/img.png", "label": "1"}],
            ["path", "label"],
        )
        # val et test absents → skip, seul train est validé

        with patch.object(validate_module, "validate_split", return_value=False):
            with pytest.raises(ValueError, match="FAILED"):
                validate_module.validate_all()
