"""
Integration tests for src/data/pipeline.py

preprocess() et download_idnet() sont mockés car ils utilisent
des chemins hardcodés indépendants de tmp_path.
"""

import csv
from pathlib import Path
from unittest.mock import patch


def _write_fake_manifests(processed_dir: Path):
    """Simule ce que preprocess() produirait dans processed_dir."""
    processed_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        csv_path = processed_dir / f"{split}.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["path", "label"])
            writer.writeheader()
            for i in range(4):
                writer.writerow({"path": f"/fake/{split}_{i}.png", "label": i % 2})


class TestPipelineIntegration:
    def test_pipeline_creates_manifests(self, tmp_path):
        from src.data.pipeline import run_pipeline

        processed_dir = tmp_path / "processed"

        with (
            patch("src.data.pipeline.download_idnet"),
            patch(
                "src.data.pipeline.preprocess",
                side_effect=lambda: _write_fake_manifests(processed_dir),
            ),
            patch("src.data.pipeline.validate_all", return_value=True),
        ):
            run_pipeline(
                raw_dir=tmp_path / "raw",
                processed_dir=processed_dir,
                skip_download=True,
            )

        for split in ("train", "val", "test"):
            assert (
                processed_dir / f"{split}.csv"
            ).exists(), f"Missing manifest: {processed_dir / f'{split}.csv'}"

    def test_pipeline_manifests_non_empty(self, tmp_path):
        from src.data.pipeline import run_pipeline

        processed_dir = tmp_path / "processed"

        with (
            patch("src.data.pipeline.download_idnet"),
            patch(
                "src.data.pipeline.preprocess",
                side_effect=lambda: _write_fake_manifests(processed_dir),
            ),
            patch("src.data.pipeline.validate_all", return_value=True),
        ):
            run_pipeline(
                raw_dir=tmp_path / "raw",
                processed_dir=processed_dir,
                skip_download=True,
            )

        for split in ("train", "val", "test"):
            with open(processed_dir / f"{split}.csv") as f:
                rows = list(csv.DictReader(f))
            assert len(rows) > 0, f"{split}.csv is empty"

    def test_pipeline_no_data_leakage(self, tmp_path):
        from src.data.pipeline import run_pipeline

        processed_dir = tmp_path / "processed"

        with (
            patch("src.data.pipeline.download_idnet"),
            patch(
                "src.data.pipeline.preprocess",
                side_effect=lambda: _write_fake_manifests(processed_dir),
            ),
            patch("src.data.pipeline.validate_all", return_value=True),
        ):
            run_pipeline(
                raw_dir=tmp_path / "raw",
                processed_dir=processed_dir,
                skip_download=True,
            )

        seen = set()
        for split in ("train", "val", "test"):
            with open(processed_dir / f"{split}.csv") as f:
                for row in csv.DictReader(f):
                    p = row["path"]
                    assert p not in seen, f"Data leakage: {p} in multiple splits"
                    seen.add(p)

    def test_download_mock_called_when_not_skipped(self, tmp_path):
        from src.data.pipeline import run_pipeline

        processed_dir = tmp_path / "processed"

        with (
            patch("src.data.pipeline.download_idnet") as mock_dl,
            patch(
                "src.data.pipeline.preprocess",
                side_effect=lambda: _write_fake_manifests(processed_dir),
            ),
            patch("src.data.pipeline.validate_all", return_value=True),
        ):
            run_pipeline(
                raw_dir=tmp_path / "raw",
                processed_dir=processed_dir,
                skip_download=False,
            )

        mock_dl.assert_called_once()

    def test_validation_step_runs(self, tmp_path):
        from src.data.pipeline import run_pipeline

        processed_dir = tmp_path / "processed"

        with (
            patch("src.data.pipeline.download_idnet"),
            patch(
                "src.data.pipeline.preprocess",
                side_effect=lambda: _write_fake_manifests(processed_dir),
            ),
            patch("src.data.pipeline.validate_all") as mock_val,
        ):
            mock_val.return_value = True
            run_pipeline(
                raw_dir=tmp_path / "raw",
                processed_dir=processed_dir,
                skip_download=True,
            )

        mock_val.assert_called_once()
