"""
Integration tests for src/models/evaluate.py

MLflow is redirected to a local tmp directory (mlflow_local fixture).
No GPU required.
"""

import numpy as np
import pytest


@pytest.fixture()
def small_loader(split_manifests):
    """A val DataLoader with the synthetic val split."""
    from torch.utils.data import DataLoader

    from src.data.dataset import VAL_TF, IDNetDataset

    ds = IDNetDataset(split_manifests["val"], transform=VAL_TF)
    return DataLoader(ds, batch_size=2, num_workers=0, shuffle=False)


@pytest.fixture()
def trained_model():
    """Untrained EfficientNet-B0 (weights are random — we only test plumbing)."""
    from src.models.efficientnet import build_model

    m = build_model(num_classes=2)
    m.eval()
    return m


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------


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

    def test_perfect_predictor_auroc_one(self):
        from src.models.evaluate import compute_metrics

        y_true = np.array([0, 0, 1, 1])
        y_score = np.array([0.0, 0.0, 1.0, 1.0])
        metrics = compute_metrics(y_true, y_score)
        assert pytest.approx(metrics["auroc"], abs=1e-6) == 1.0

    def test_random_predictor_auroc_near_half(self):
        from src.models.evaluate import compute_metrics

        rng = np.random.default_rng(42)
        y_true = rng.integers(0, 2, size=200)
        y_score = rng.uniform(0, 1, size=200)
        metrics = compute_metrics(y_true, y_score)
        # Random predictor AUROC should be ≈ 0.5 ± 0.15
        assert 0.35 <= metrics["auroc"] <= 0.65


# ---------------------------------------------------------------------------
# Evaluate on loader
# ---------------------------------------------------------------------------


class TestEvaluateOnLoader:
    @pytest.mark.slow
    def test_evaluate_returns_dict(self, trained_model, small_loader, mlflow_local):
        from src.models.evaluate import evaluate

        metrics = evaluate(trained_model, small_loader)
        assert isinstance(metrics, dict)
        assert "auroc" in metrics

    @pytest.mark.slow
    def test_evaluate_logs_to_mlflow(self, trained_model, small_loader, mlflow_local):
        import mlflow

        from src.models.evaluate import evaluate

        # 1. On capture l'objet 'run' ici
        with mlflow.start_run() as run:
            # 2. On vérifie que run n'est pas None (sécurité pour le linter)
            if run is None:
                pytest.fail("MLflow n'a pas pu démarrer le run.")

            # 3. Exécution de l'évaluation
            evaluate(trained_model, small_loader, log_to_mlflow=True)

            # 4. On utilise 'run.info.run_id' directement au lieu de active_run()
            client = mlflow.tracking.MlflowClient()
            run_data = client.get_run(run.info.run_id).data
            logged_metrics = run_data.metrics

        # 5. Assertions
        assert "test_auroc" in logged_metrics or "auroc" in logged_metrics

    @pytest.mark.slow
    def test_evaluate_saves_confusion_matrix(
        self, trained_model, small_loader, mlflow_local, tmp_path
    ):
        from src.models.evaluate import evaluate

        metrics = evaluate(
            trained_model,
            small_loader,
            artifact_dir=tmp_path,
        )
        # Either the function returns an artefact path or saves a file
        cm_files = list(tmp_path.glob("*confusion*")) + list(tmp_path.glob("*cm*"))
        # Loose check: either files exist OR the key is in metrics
        assert (
            len(cm_files) > 0 or "confusion_matrix" in metrics
        ), "Confusion matrix artefact should be produced"


# ---------------------------------------------------------------------------
# Quality gate integration
# ---------------------------------------------------------------------------


class TestQualityGateIntegration:
    def test_evaluate_raises_on_low_auroc(self, trained_model, small_loader, mlflow_local):
        """
        If the model fails the quality gate, evaluate() should raise or
        return a failing flag — depending on implementation.
        """
        import unittest.mock as mock

        from src.models.config import MIN_AUROC
        from src.models.evaluate import evaluate

        # Force AUROC below threshold
        with mock.patch("src.models.evaluate.compute_metrics") as mock_metrics:
            mock_metrics.return_value = {
                "auroc": MIN_AUROC - 0.05,
                "accuracy": 0.5,
            }
            try:
                result = evaluate(trained_model, small_loader, enforce_gate=True)
                # If it doesn't raise, it must return a failure indicator
                passed = result.get("gate_passed", True)
                assert not passed, "evaluate() must signal failure when AUROC < MIN_AUROC"
            except (ValueError, RuntimeError, SystemExit):
                pass  # Raising is also acceptable behaviour
