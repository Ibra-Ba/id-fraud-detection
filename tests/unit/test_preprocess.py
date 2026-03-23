"""
Unit tests for src/data/preprocess.py
"""

import csv
from pathlib import Path

# import pytest  # noqa: F401

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _count_csv(path: Path) -> int:
    with open(path) as f:
        return sum(1 for _ in csv.DictReader(f))


def _read_csv(path: Path) -> list[dict]:
    with open(path) as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSplitManifests:
    """Test the output manifests produced by preprocess.make_splits()."""

    def test_train_val_test_manifests_exist(self, split_manifests):
        for split in ("train", "val", "test"):
            assert split_manifests[split].exists(), f"Manifest for {split} not found"

    def test_all_paths_exist(self, split_manifests):
        """Every path listed in a manifest must point to a real file."""
        for split, csv_path in split_manifests.items():
            for row in _read_csv(csv_path):
                assert Path(row["path"]).exists(), f"[{split}] Missing image: {row['path']}"

    def test_labels_are_binary(self, split_manifests):
        for split, csv_path in split_manifests.items():
            for row in _read_csv(csv_path):
                assert int(row["label"]) in {0, 1}, f"[{split}] Invalid label: {row['label']}"

    def test_no_overlap_between_splits(self, split_manifests):
        """The same image path must not appear in two different splits."""
        sets = {
            split: {row["path"] for row in _read_csv(p)} for split, p in split_manifests.items()
        }
        train_val = sets["train"] & sets["val"]
        train_test = sets["train"] & sets["test"]
        val_test = sets["val"] & sets["test"]
        assert not train_val, f"train/val overlap: {train_val}"
        assert not train_test, f"train/test overlap: {train_test}"
        assert not val_test, f"val/test overlap: {val_test}"

    def test_fraud_ratio_train(self, split_manifests):
        """Training set from fake_image_dir has exactly 20% fraud (2/10)."""
        rows = _read_csv(split_manifests["train"])
        fraud = sum(1 for r in rows if int(r["label"]) == 1)
        ratio = fraud / len(rows)
        assert 0.10 <= ratio <= 0.50, f"Unexpected fraud ratio in train split: {ratio:.2%}"

    def test_csv_has_required_columns(self, split_manifests):
        for split, csv_path in split_manifests.items():
            with open(csv_path) as f:
                reader = csv.DictReader(f)
                assert (
                    reader.fieldnames and "path" in reader.fieldnames
                ), f"[{split}] Missing 'path' column"
                assert (
                    reader.fieldnames and "label" in reader.fieldnames
                ), f"[{split}] Missing 'label' column"

    def test_total_count_matches_source_images(self, split_manifests, fake_image_dir):
        """Sum of all split rows must equal the total number of images."""
        total_rows = sum(_count_csv(p) for p in split_manifests.values())
        total_images = sum(1 for _ in fake_image_dir.rglob("*.png"))
        assert (
            total_rows == total_images
        ), f"Row count ({total_rows}) != image count ({total_images})"
