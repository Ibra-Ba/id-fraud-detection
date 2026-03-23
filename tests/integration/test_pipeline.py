"""
Integration tests for src/data/pipeline.py

The download step is mocked (no network).
The preprocess + validate steps run on synthetic data in tmp_path.
"""

from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _populate_raw_dir(raw_dir: Path, n_genuine: int = 20, n_fraud: int = 5):
    """
    Create a minimal IDNet-like directory structure in *raw_dir*.

        raw_dir/
          genuine/  ← n_genuine PNGs
          fraud/    ← n_fraud PNGs
    """
    for label in ("genuine", "fraud"):
        folder = raw_dir / label
        folder.mkdir(parents=True)
    for i in range(n_genuine):
        arr = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        Image.fromarray(arr).save(raw_dir / "genuine" / f"g_{i}.png")
    for i in range(n_fraud):
        arr = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        Image.fromarray(arr).save(raw_dir / "fraud" / f"f_{i}.png")


# ---------------------------------------------------------------------------
# Pipeline integration tests
# ---------------------------------------------------------------------------


class TestPipelineIntegration:
    def test_pipeline_creates_manifests(self, tmp_path):
        """
        Full pipeline (download mocked) should produce train/val/test CSVs.
        """
        from src.data.pipeline import run_pipeline

        raw_dir = tmp_path / "raw"
        processed_dir = tmp_path / "processed"

        # Simulate a successful download by pre-populating the raw dir
        _populate_raw_dir(raw_dir)

        with patch("src.data.pipeline.download_idnet") as mock_dl:
            mock_dl.return_value = raw_dir
            run_pipeline(
                raw_dir=raw_dir,
                processed_dir=processed_dir,
                skip_download=True,
            )

        for split in ("train", "val", "test"):
            manifest = processed_dir / f"{split}.csv"
            assert manifest.exists(), f"Missing manifest: {manifest}"

    def test_pipeline_manifests_non_empty(self, tmp_path):
        from src.data.pipeline import run_pipeline

        raw_dir = tmp_path / "raw"
        processed_dir = tmp_path / "processed"
        _populate_raw_dir(raw_dir)

        with patch("src.data.pipeline.download_idnet") as mock_dl:
            mock_dl.return_value = raw_dir
            run_pipeline(raw_dir=raw_dir, processed_dir=processed_dir, skip_download=True)

        import csv

        for split in ("train", "val", "test"):
            manifest = processed_dir / f"{split}.csv"
            with open(manifest) as f:
                rows = list(csv.DictReader(f))
            assert len(rows) > 0, f"{split}.csv is empty"

    def test_pipeline_no_data_leakage(self, tmp_path):
        """Images must not appear in more than one split."""
        import csv

        from src.data.pipeline import run_pipeline

        raw_dir = tmp_path / "raw"
        processed_dir = tmp_path / "processed"
        _populate_raw_dir(raw_dir, n_genuine=30, n_fraud=10)

        with patch("src.data.pipeline.download_idnet") as mock_dl:
            mock_dl.return_value = raw_dir
            run_pipeline(raw_dir=raw_dir, processed_dir=processed_dir, skip_download=True)

        seen = set()
        for split in ("train", "val", "test"):
            with open(processed_dir / f"{split}.csv") as f:
                for row in csv.DictReader(f):
                    p = row["path"]
                    assert p not in seen, f"Data leakage: {p} in multiple splits"
                    seen.add(p)

    def test_download_mock_called_when_not_skipped(self, tmp_path):
        from src.data.pipeline import run_pipeline

        raw_dir = tmp_path / "raw"
        processed_dir = tmp_path / "processed"
        _populate_raw_dir(raw_dir)

        with patch("src.data.pipeline.download_idnet") as mock_dl:
            mock_dl.return_value = raw_dir
            run_pipeline(raw_dir=raw_dir, processed_dir=processed_dir, skip_download=False)

        mock_dl.assert_called_once()

    def test_validation_step_runs(self, tmp_path):
        """Pipeline should invoke validate_all."""
        from src.data.pipeline import run_pipeline

        # 1. IL FAUT DEFINIR LES CHEMINS ICI AUSSI
        raw_dir = tmp_path / "raw"
        processed_dir = tmp_path / "processed"
        _populate_raw_dir(raw_dir)

        with (
            patch("src.data.pipeline.download_idnet") as mock_dl,
            patch("src.data.pipeline.validate_all") as mock_val,
        ):

            mock_dl.return_value = None
            mock_val.return_value = True

            run_pipeline(raw_dir=raw_dir, processed_dir=processed_dir, skip_download=True)

        assert mock_val.call_count >= 1, "validate_all n'a pas été appelée !"
