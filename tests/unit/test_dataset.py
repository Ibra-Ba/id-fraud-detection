"""
Unit tests for src/data/dataset.py — IDNetDataset and transforms.
No real IDNet data required: all images are synthetic.
"""

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

# IDNetDataset — basic contract


class TestIDNetDataset:
    def test_len(self, split_manifests, tmp_path):
        from src.data.dataset import VAL_TF, IDNetDataset

        ds = IDNetDataset(split_manifests["train"], transform=VAL_TF)

        assert len(ds) == 10

    def test_getitem_shapes(self, split_manifests):
        from src.data.dataset import VAL_TF, IDNetDataset

        ds = IDNetDataset(split_manifests["val"], transform=VAL_TF)
        image, label = ds[0]
        assert isinstance(image, torch.Tensor)
        assert image.shape == (3, 224, 224), f"Expected (3, 224, 224), got {image.shape}"

    def test_getitem_label_dtype(self, split_manifests):
        from src.data.dataset import VAL_TF, IDNetDataset

        ds = IDNetDataset(split_manifests["train"], transform=VAL_TF)
        _, label = ds[0]
        assert isinstance(label, (int, torch.Tensor))

    def test_label_values_binary(self, split_manifests):
        from src.data.dataset import VAL_TF, IDNetDataset

        ds = IDNetDataset(split_manifests["train"], transform=VAL_TF)
        labels = {int(ds[i][1]) for i in range(len(ds))}
        assert labels.issubset({0, 1}), f"Found non-binary labels: {labels}"

    def test_no_nan_in_tensor(self, split_manifests):
        from src.data.dataset import VAL_TF, IDNetDataset

        ds = IDNetDataset(split_manifests["val"], transform=VAL_TF)
        image, _ = ds[0]
        assert not torch.isnan(image).any(), "Image tensor contains NaN"

    def test_pixel_range_after_normalize(self, split_manifests):
        """After ImageNet normalisation, most values should be in [-3, 3]."""
        from src.data.dataset import VAL_TF, IDNetDataset

        ds = IDNetDataset(split_manifests["val"], transform=VAL_TF)
        image, _ = ds[0]
        assert image.min() > -4.0
        assert image.max() < 4.0


# Transforms


class TestTransforms:
    def test_train_transform_output_shape(self):
        from src.data.dataset import TRAIN_TF

        pil_img = Image.fromarray(np.random.randint(0, 256, (300, 400, 3), dtype=np.uint8))
        result = TRAIN_TF(image=np.array(pil_img))
        tensor = result["image"] if isinstance(result, dict) else result
        assert tensor.shape == (3, 224, 224)

    def test_val_transform_output_shape(self):
        from src.data.dataset import VAL_TF

        pil_img = Image.fromarray(np.random.randint(0, 256, (300, 400, 3), dtype=np.uint8))
        result = VAL_TF(image=np.array(pil_img))
        tensor = result["image"] if isinstance(result, dict) else result
        assert tensor.shape == (3, 224, 224)

    def test_train_transform_is_not_deterministic(self):
        """Training transforms include random augmentations.
        On tente 10 fois — la proba que toutes soient identiques est négligeable.
        """
        from src.data.dataset import TRAIN_TF

        arr = np.random.randint(0, 256, (300, 400, 3), dtype=np.uint8)

        def _apply():
            r = TRAIN_TF(image=arr.copy())
            return r["image"] if isinstance(r, dict) else r

        results = [_apply() for _ in range(10)]
        all_equal = all(torch.equal(results[0], r) for r in results[1:])
        assert not all_equal, "TRAIN_TF should be stochastic — all 10 outputs were identical"

    def test_val_transform_is_deterministic(self):
        """Validation/test transforms must be deterministic."""
        from src.data.dataset import VAL_TF

        arr = np.random.randint(0, 256, (300, 400, 3), dtype=np.uint8)

        def _apply(arr):
            r = VAL_TF(image=arr.copy())
            return r["image"] if isinstance(r, dict) else r

        t1 = _apply(arr)
        t2 = _apply(arr)
        assert torch.equal(t1, t2), "VAL_TF must be deterministic"


# ---------------------------------------------------------------------------
# DataLoader integration
# ---------------------------------------------------------------------------


class TestDataLoader:
    def test_dataloader_batch(self, split_manifests):
        from src.data.dataset import VAL_TF, IDNetDataset

        ds = IDNetDataset(split_manifests["train"], transform=VAL_TF)
        loader = DataLoader(ds, batch_size=4, num_workers=0, shuffle=False)
        images, labels = next(iter(loader))
        assert images.shape == (4, 3, 224, 224)
        assert labels.shape == (4,)

    def test_dataloader_num_workers_zero(self, split_manifests):
        """Explicit guard: num_workers must stay 0 for WSL2."""
        from src.data.dataset import VAL_TF, IDNetDataset

        ds = IDNetDataset(split_manifests["train"], transform=VAL_TF)
        loader = DataLoader(ds, batch_size=2, num_workers=0)
        # should not raise
        _ = next(iter(loader))
