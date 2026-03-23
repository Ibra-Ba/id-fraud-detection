"""
Unit tests for src/models/train.py

Strategy: no real training loop is run. We test the individual pieces:
  - forward pass produces finite loss
  - quality gate logic (AUROC threshold)
  - Phase 1 / Phase 2 param-group configuration
"""

import pytest
import torch
import torch.nn as nn


@pytest.fixture()
def tiny_model():
    from src.models.efficientnet import FraudClassifier

    return FraudClassifier(num_classes=2)


@pytest.fixture()
def tiny_batch():
    """Minimal (2, 3, 224, 224) batch with binary labels."""
    images = torch.randn(2, 3, 224, 224)
    labels = torch.tensor([0, 1], dtype=torch.long)
    return images, labels


# ---------------------------------------------------------------------------
# Forward pass & loss
# ---------------------------------------------------------------------------


class TestForwardPass:
    def test_loss_is_finite(self, tiny_model, tiny_batch):
        tiny_model.train()
        images, labels = tiny_batch
        criterion = nn.CrossEntropyLoss()
        logits = tiny_model(images)
        loss = criterion(logits, labels)
        assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"

    def test_loss_is_scalar(self, tiny_model, tiny_batch):
        tiny_model.train()
        images, labels = tiny_batch
        criterion = nn.CrossEntropyLoss()
        logits = tiny_model(images)
        loss = criterion(logits, labels)
        assert loss.ndim == 0, "CrossEntropyLoss should return a scalar"

    def test_backward_does_not_raise(self, tiny_model, tiny_batch):
        from src.models.efficientnet import freeze_backbone

        freeze_backbone(tiny_model)  # Phase 1: only head is trainable
        tiny_model.train()
        images, labels = tiny_batch
        criterion = nn.CrossEntropyLoss()
        logits = tiny_model(images)
        loss = criterion(logits, labels)
        loss.backward()  # must not raise

    def test_logits_shape(self, tiny_model, tiny_batch):
        tiny_model.eval()
        images, _ = tiny_batch
        with torch.no_grad():
            logits = tiny_model(images)
        assert logits.shape == (2, 2)


# ---------------------------------------------------------------------------
# Quality gate
# ---------------------------------------------------------------------------


class TestQualityGate:
    def test_gate_passes_above_threshold(self):
        from src.models.config import MIN_AUROC
        from src.models.train import check_quality_gate

        assert check_quality_gate(auroc=MIN_AUROC + 0.01) is True

    def test_gate_passes_at_exact_threshold(self):
        from src.models.config import MIN_AUROC
        from src.models.train import check_quality_gate

        assert check_quality_gate(auroc=MIN_AUROC) is True

    def test_gate_fails_below_threshold(self):
        from src.models.config import MIN_AUROC
        from src.models.train import check_quality_gate

        assert check_quality_gate(auroc=MIN_AUROC - 0.01) is False

    def test_gate_raises_on_nan(self):
        from src.models.train import check_quality_gate

        with pytest.raises((ValueError, AssertionError)):
            check_quality_gate(auroc=float("nan"))


# ---------------------------------------------------------------------------
# Optimizer param groups (phase detection)
# ---------------------------------------------------------------------------


class TestOptimizerConfig:
    def test_phase1_only_head_params(self, tiny_model):
        """In phase 1, the optimiser should only receive classifier params."""
        from src.models.efficientnet import freeze_backbone
        from src.models.train import make_optimizer

        freeze_backbone(tiny_model)
        optimizer = make_optimizer(tiny_model, lr=1e-3, phase=1)

        # On récupère les paramètres de l'optimiseur
        optimizer_params = [p for group in optimizer.param_groups for p in group["params"]]

        # 1. On vérifie que l'optimiseur n'est pas vide
        assert len(optimizer_params) > 0, "L'optimiseur ne contient aucun paramètre !"

        # 2. On vérifie que CHAQUE paramètre dans l'optimiseur est bien entraînable
        for p in optimizer_params:
            assert (
                p.requires_grad is True
            ), "L'optimiseur de Phase 1 contient un paramètre gelé (requires_grad=False) !"

    def test_phase2_all_params(self, tiny_model):
        """Phase 2 unfreezes everything; optimiser covers all params."""
        from src.models.efficientnet import unfreeze_backbone
        from src.models.train import make_optimizer

        unfreeze_backbone(tiny_model)
        optimizer = make_optimizer(tiny_model, lr=1e-4, phase=2)
        n_optimizer_params = sum(
            p.numel() for group in optimizer.param_groups for p in group["params"]
        )
        n_model_params = sum(p.numel() for p in tiny_model.parameters())
        assert (
            n_optimizer_params == n_model_params
        ), "Phase-2 optimizer should cover all model parameters"
