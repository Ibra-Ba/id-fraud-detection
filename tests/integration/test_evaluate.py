"""
Integration tests for src/models/evaluate.py
"""

import numpy as np
import pytest


@pytest.fixture()
def small_loader(split_manifests):
    from torch.utils.data import DataLoader

    from src.data.dataset import VAL_TF, IDNetDataset

    ds = IDNetDataset(split_manifests["val"], transform=VAL_TF)
    return DataLoader(ds, batch_size=2, num_workers=0, shuffle=False)


@pytest.fixture()
def trained_model():
    from src.models.efficientnet import FraudClassifier

    m = FraudClassifier(pretrained=False)
    m.eval()
    return m


class TestComputeMetrics:
    def test_auroc_in_range(self):
        from src.models.evaluate import compute_metrics

        y_true = np.array([0, 0, 1, 1])
        y_score = np.array([0.1, 0.4, 0.6, 0.9])
        metrics = compute_metrics(y_true, y_score)
        assert "auroc" in metrics
        assert 0.0 <= metrics["auroc"] <= 1.0

    def test_accuracy_in_range(self):
        from src.models.evaluate import compute_metrics

        y_true = np.array([0, 1, 0, 1])
        y_score = np.array([0.2, 0.8, 0.3, 0.7])
        metrics = compute_metrics(y_true, y_score)
        assert "accuracy" in metrics
        assert 0.0 <= metrics["accuracy"] <= 1.0

    def test_f1_key_exists(self):
        from src.models.evaluate import compute_metrics

        y_true = np.array([0, 1, 0, 1])
        y_score = np.array([0.2, 0.8, 0.3, 0.7])
        metrics = compute_metrics(y_true, y_score)
        assert "f1" in metrics, f"Expected key 'f1', got: {list(metrics.keys())}"

    def test_perfect_predictor_auroc_one(self):
        from src.models.evaluate import compute_metrics

        y_true = np.array([0, 0, 1, 1])
        y_score = np.array([0.0, 0.0, 1.0, 1.0])
        metrics = compute_metrics(y_true, y_score)
        assert pytest.approx(metrics["auroc"], abs=1e-6) == 1.0


class TestQualityGateIntegration:
    def test_evaluate_fails_on_low_auroc(self, trained_model, small_loader, mlflow_local):
        """Gate doit échouer quand AUROC < MIN_AUROC."""
        from unittest.mock import patch

        from src.models.config import MIN_AUROC
        from src.models.evaluate import evaluate

        low_metrics = {
            "auroc": MIN_AUROC - 0.05,
            "accuracy": 0.5,
            "f1": 0.4,
            "threshold_used": 0.5,
        }

        with patch("src.models.evaluate.compute_metrics", return_value=low_metrics):
            result = evaluate(trained_model, small_loader, enforce_gate=True)

        passed = result.get("gate_passed", True)
        assert not passed, "gate_passed doit être False quand AUROC < MIN_AUROC"

    def test_evaluate_passes_on_high_auroc(self, trained_model, small_loader, mlflow_local):
        """Gate doit passer quand AUROC >= MIN_AUROC."""
        from unittest.mock import patch

        from src.models.config import MIN_AUROC
        from src.models.evaluate import evaluate

        high_metrics = {
            "auroc": MIN_AUROC + 0.05,
            "accuracy": 0.9,
            "f1": 0.9,
            "threshold_used": 0.5,
        }

        with patch("src.models.evaluate.compute_metrics", return_value=high_metrics):
            result = evaluate(trained_model, small_loader, enforce_gate=True)

        assert result.get("gate_passed") is True
