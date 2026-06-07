"""
Unit tests for src/training/retrain_pipeline.py
"""

from unittest.mock import MagicMock, patch

import pytest

# ── check_champion_health ─────────────────────────────────────────────────────


class TestCheckChampionHealth:
    def _make_client(self, auroc=0.97, threshold=0.1014, recall=0.9505):
        """Construit un MlflowClient mock avec un champion valide."""
        client = MagicMock()

        mv = MagicMock()
        mv.version = "3"
        mv.run_id = "fake-run-id-abc"
        client.get_model_version_by_alias.return_value = mv

        run = MagicMock()
        run.data.metrics = {
            "test_auroc": auroc,
            "optimal_threshold": threshold,
            "test_recall": recall,
        }
        run.data.tags = {"origin": "bloc6_training"}
        client.get_run.return_value = run

        return client

    def test_returns_dict_with_required_keys(self):
        from src.training.retrain_pipeline import check_champion_health

        client = self._make_client()
        result = check_champion_health(client)

        for key in ("run_id", "version", "auroc", "threshold", "recall", "healthy", "origin"):
            assert key in result, f"Clé manquante : {key}"

    def test_healthy_when_auroc_above_min(self):
        from src.training.retrain_pipeline import MIN_AUROC, check_champion_health

        client = self._make_client(auroc=MIN_AUROC + 0.01)
        result = check_champion_health(client)
        assert result["healthy"] is True

    def test_unhealthy_when_auroc_below_min(self):
        from src.training.retrain_pipeline import MIN_AUROC, check_champion_health

        client = self._make_client(auroc=MIN_AUROC - 0.05)
        result = check_champion_health(client)
        assert result["healthy"] is False

    def test_threshold_from_metrics_first(self):
        from src.training.retrain_pipeline import check_champion_health

        client = self._make_client(threshold=0.1014)
        result = check_champion_health(client)
        assert abs(result["threshold"] - 0.1014) < 1e-6

    def test_threshold_fallback_to_tag(self):
        """Si optimal_threshold absent des metrics → lire le tag."""
        from src.training.retrain_pipeline import check_champion_health

        client = MagicMock()
        mv = MagicMock()
        mv.version = "2"
        mv.run_id = "run-tag-fallback"
        client.get_model_version_by_alias.return_value = mv

        run = MagicMock()
        run.data.metrics = {"test_auroc": 0.97}  # pas d'optimal_threshold
        run.data.tags = {"optimal_threshold": "0.2500", "origin": "bloc6_training"}
        client.get_run.return_value = run

        result = check_champion_health(client)
        assert abs(result["threshold"] - 0.25) < 1e-6

    def test_raises_when_no_champion(self):
        from src.training.retrain_pipeline import check_champion_health

        client = MagicMock()
        client.get_model_version_by_alias.side_effect = Exception("No alias found")

        with pytest.raises(RuntimeError, match="Champion introuvable"):
            check_champion_health(client)

    def test_recall_none_when_not_in_metrics(self):
        """recall peut être absent — ne doit pas faire planter."""
        from src.training.retrain_pipeline import check_champion_health

        client = MagicMock()
        mv = MagicMock()
        mv.version = "1"
        mv.run_id = "run-no-recall"
        client.get_model_version_by_alias.return_value = mv

        run = MagicMock()
        run.data.metrics = {"test_auroc": 0.97, "optimal_threshold": 0.10}
        run.data.tags = {"origin": "bloc6_training"}
        client.get_run.return_value = run

        result = check_champion_health(client)
        assert result["recall"] is None

    def test_version_and_run_id_in_result(self):
        from src.training.retrain_pipeline import check_champion_health

        client = self._make_client()
        result = check_champion_health(client)

        assert result["version"] == "3"
        assert result["run_id"] == "fake-run-id-abc"


# ── write_github_outputs ──────────────────────────────────────────────────────


class TestWriteGithubOutputs:
    def test_writes_required_keys(self, tmp_path, monkeypatch):
        from src.training.retrain_pipeline import write_github_outputs

        output_file = tmp_path / "github_output"
        output_file.write_text("")
        monkeypatch.setenv("GITHUB_OUTPUT", str(output_file))

        health = {
            "run_id": "run-abc",
            "auroc": 0.9587,
            "threshold": 0.1014,
            "version": "3",
            "healthy": True,
        }
        write_github_outputs(health)

        content = output_file.read_text()
        assert "new_run_id=run-abc" in content
        assert "new_auroc=0.9587" in content
        assert "champion_auroc=0.9587" in content
        assert "promoted=false" in content

    def test_no_crash_when_github_output_not_set(self, monkeypatch):
        from src.training.retrain_pipeline import write_github_outputs

        monkeypatch.delenv("GITHUB_OUTPUT", raising=False)

        health = {"run_id": "x", "auroc": 0.95, "version": "1", "healthy": True}
        write_github_outputs(health)  # ne doit pas lever


# ── log_ct_run ────────────────────────────────────────────────────────────────


class TestLogCtRun:
    def test_logs_to_monitoring_experiment(self, mlflow_local):
        import mlflow

        from src.training.retrain_pipeline import log_ct_run

        health = {
            "run_id": "run-abc",
            "version": "3",
            "auroc": 0.9587,
            "threshold": 0.1014,
            "recall": 0.9505,
            "healthy": True,
            "origin": "bloc6_training",
        }

        ct_run_id = log_ct_run(health, reason="drift detected")

        # Vérifie que le run existe dans l'expérience monitoring
        client = mlflow.tracking.MlflowClient()
        run = client.get_run(ct_run_id)
        assert run.data.tags["ct_mode"] == "stub_bac5"
        assert run.data.tags["trigger_reason"] == "drift detected"
        assert run.data.tags["champion_healthy"] == "True"

    def test_logs_auroc_metric(self, mlflow_local):
        import mlflow

        from src.training.retrain_pipeline import log_ct_run

        health = {
            "run_id": "run-xyz",
            "version": "2",
            "auroc": 0.9600,
            "threshold": 0.12,
            "recall": None,
            "healthy": True,
            "origin": "bloc6_training",
        }

        ct_run_id = log_ct_run(health, reason="manual")

        client = mlflow.tracking.MlflowClient()
        run = client.get_run(ct_run_id)
        assert abs(run.data.metrics["champion_auroc"] - 0.9600) < 1e-4


# ── main — exit codes ─────────────────────────────────────────────────────────


class TestMainExitCodes:
    def test_exits_zero_when_healthy(self, mlflow_local, monkeypatch, tmp_path):
        from src.training.retrain_pipeline import MIN_AUROC

        monkeypatch.setenv("GITHUB_OUTPUT", str(tmp_path / "out"))
        (tmp_path / "out").write_text("")

        healthy = {
            "run_id": "r",
            "version": "1",
            "auroc": MIN_AUROC + 0.01,
            "threshold": 0.1,
            "recall": 0.96,
            "healthy": True,
            "origin": "bloc6_training",
        }

        with (
            patch("src.training.retrain_pipeline.check_champion_health", return_value=healthy),
            patch("src.training.retrain_pipeline.log_ct_run", return_value="fake-run"),
        ):
            from src.training.retrain_pipeline import main

            result = main()

        assert result["healthy"] is True

    def test_exits_two_when_degraded(self, mlflow_local, monkeypatch, tmp_path):
        from src.training.retrain_pipeline import MIN_AUROC

        monkeypatch.setenv("GITHUB_OUTPUT", str(tmp_path / "out"))
        (tmp_path / "out").write_text("")

        degraded = {
            "run_id": "r",
            "version": "1",
            "auroc": MIN_AUROC - 0.05,
            "threshold": 0.1,
            "recall": 0.80,
            "healthy": False,
            "origin": "bloc6_training",
        }

        with (
            patch("src.training.retrain_pipeline.check_champion_health", return_value=degraded),
            patch("src.training.retrain_pipeline.log_ct_run", return_value="fake-run"),
        ):
            from src.training.retrain_pipeline import main

            with pytest.raises(SystemExit) as exc_info:
                main()

        assert exc_info.value.code == 2
