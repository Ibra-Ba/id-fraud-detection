"""
Unit tests for src/models/efficientnet.py
"""

import pytest
import torch


@pytest.fixture()
def model():
    from src.models.efficientnet import build_model

    return build_model(num_classes=2)


class TestBuildModel:
    def test_returns_nn_module(self, model):
        import torch.nn as nn

        assert isinstance(model, nn.Module)

    def test_output_shape(self, model):
        model.eval()
        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, 2), f"Expected (2, 2), got {out.shape}"

    def test_output_no_nan(self, model):
        model.eval()
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            out = model(x)
        assert not torch.isnan(out).any()

    def test_default_num_classes_two(self):
        from src.models.efficientnet import build_model

        m = build_model(num_classes=2)
        m.eval()
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            out = m(x)
        assert out.shape[-1] == 2


class TestFreezeBackbone:
    def test_backbone_params_frozen(self, model):
        from src.models.efficientnet import freeze_backbone

        freeze_backbone(model)
        trainable = [
            n for n, p in model.named_parameters() if "classifier" not in n and p.requires_grad
        ]
        assert (
            len(trainable) == 0
        ), f"Expected backbone frozen, but these params are still trainable: {trainable}"

    def test_classifier_params_trainable_after_freeze(self, model):
        from src.models.efficientnet import freeze_backbone

        freeze_backbone(model)
        trainable = [
            n for n, p in model.named_parameters() if "classifier" in n and p.requires_grad
        ]
        assert len(trainable) > 0, "Classifier head must remain trainable after freeze_backbone()"

    def test_forward_pass_after_freeze(self, model):
        from src.models.efficientnet import freeze_backbone

        freeze_backbone(model)
        model.eval()
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 2)


class TestUnfreezeBackbone:
    def test_all_params_trainable_after_unfreeze(self, model):
        from src.models.efficientnet import freeze_backbone, unfreeze_backbone

        freeze_backbone(model)
        unfreeze_backbone(model)
        frozen = [n for n, p in model.named_parameters() if not p.requires_grad]
        assert len(frozen) == 0, f"Expected all params trainable, but these are frozen: {frozen}"

    def test_gradient_flows_after_unfreeze(self, model):
        from src.models.efficientnet import freeze_backbone, unfreeze_backbone

        freeze_backbone(model)
        unfreeze_backbone(model)
        model.train()
        x = torch.randn(1, 3, 224, 224)
        out = model(x)
        loss = out.sum()
        loss.backward()
        # At least one backbone parameter should have a non-None gradient
        backbone_grads = [
            p.grad
            for n, p in model.named_parameters()
            if "classifier" not in n and p.grad is not None
        ]
        assert len(backbone_grads) > 0, "No gradients flowed through backbone after unfreeze"
