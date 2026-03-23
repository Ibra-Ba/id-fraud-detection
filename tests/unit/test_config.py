"""
Unit tests for src/models/config.py
"""

# import torch


def test_device_is_cpu():
    from src.models.config import DEVICE

    # assert str(DEVICE) == torch.device("cpu"), "DEVICE must be 'cpu' for the WSL2/CPU-only constraint"
    assert str(DEVICE) == "cpu", "DEVICE must be 'cpu' for the WSL2/CPU-only constraint"


def test_batch_size_positive():
    from src.models.config import BATCH_SIZE

    assert isinstance(BATCH_SIZE, int)
    assert BATCH_SIZE > 0


def test_min_auroc_range():
    from src.models.config import MIN_AUROC

    assert 0.0 < MIN_AUROC <= 1.0, "MIN_AUROC must be in (0, 1]"


def test_num_classes():
    from src.models.config import NUM_CLASSES

    assert NUM_CLASSES == 2, "Binary classifier: genuine vs fraud"


def test_image_size_positive():
    from src.models.config import IMAGE_SIZE

    assert isinstance(IMAGE_SIZE, int)
    assert IMAGE_SIZE > 0


def test_num_workers_zero():
    """num_workers=0 is required for WSL2 DataLoader compatibility."""
    from src.models.config import NUM_WORKERS

    assert NUM_WORKERS == 0, "NUM_WORKERS must be 0 under WSL2 to avoid multiprocessing issues"
