"""
Shared fixtures for the id-fraud-detector test suite.

All fixtures are purely synthetic — no IDNet data required.
"""

import csv
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

# ---------------------------------------------------------------------------
# Pytest markers
# ---------------------------------------------------------------------------


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow-running")
    config.addinivalue_line("markers", "integration: mark test as integration-level")


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------


def _make_fake_image(path: Path, size: tuple[int, int] = (224, 224)) -> Path:
    """Write a random RGB PNG to *path* and return it."""
    arr = np.random.randint(0, 256, (*size, 3), dtype=np.uint8)
    Image.fromarray(arr).save(path)
    return path


# ---------------------------------------------------------------------------
# Dataset / manifest fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def fake_image(tmp_path) -> Path:
    """Single 224×224 RGB image."""
    return _make_fake_image(tmp_path / "img.png")


@pytest.fixture()
def fake_image_dir(tmp_path) -> Path:
    """
    Directory tree mimicking the IDNet split layout:

        root/
          train/genuine/  (8 images)
          train/fraud/    (2 images)
          val/genuine/    (2 images)
          val/fraud/      (1 image)
          test/genuine/   (2 images)
          test/fraud/     (1 image)
    """
    counts = {
        ("train", "genuine"): 8,
        ("train", "fraud"): 2,
        ("val", "genuine"): 2,
        ("val", "fraud"): 1,
        ("test", "genuine"): 2,
        ("test", "fraud"): 1,
    }
    for (split, label), n in counts.items():
        folder = tmp_path / split / label
        folder.mkdir(parents=True)
        for i in range(n):
            _make_fake_image(folder / f"{split}_{label}_{i}.png")
    return tmp_path


@pytest.fixture()
def fake_manifest_csv(tmp_path, fake_image_dir) -> Path:
    """
    CSV manifest in the format produced by preprocess.py:
        path, label
    """
    csv_path = tmp_path / "manifest.csv"
    rows = []
    for split in ("train", "val", "test"):
        for label_str, label_int in (("genuine", 0), ("fraud", 1)):
            folder = fake_image_dir / split / label_str
            for img in folder.iterdir():
                rows.append({"path": str(img), "label": label_int})

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "label"])
        writer.writeheader()
        writer.writerows(rows)

    return csv_path


@pytest.fixture()
def split_manifests(tmp_path, fake_image_dir) -> dict[str, Path]:
    """
    Returns {"train": Path, "val": Path, "test": Path} — one CSV per split.
    """
    manifests = {}
    for split in ("train", "val", "test"):
        csv_path = tmp_path / f"{split}.csv"
        rows = []
        for label_str, label_int in (("genuine", 0), ("fraud", 1)):
            folder = fake_image_dir / split / label_str
            for img in folder.iterdir():
                rows.append({"path": str(img), "label": label_int})
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["path", "label"])
            writer.writeheader()
            writer.writerows(rows)
        manifests[split] = csv_path
    return manifests


# ---------------------------------------------------------------------------
# MLflow fixture (in-memory, no remote tracking)
# ---------------------------------------------------------------------------


@pytest.fixture()
def mlflow_local(tmp_path, monkeypatch):
    """
    Redirect MLflow to a local tmp directory so tests never hit the HF Space.
    Returns the tracking URI string.
    """
    import mlflow

    tracking_uri = f"file://{tmp_path / 'mlruns'}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking_uri)
    mlflow.set_tracking_uri(tracking_uri)
    return tracking_uri
