"""
Unit tests for src/models/simulate_threshold.py
"""

import numpy as np
import pytest

# ── find_optimal_threshold ────────────────────────────────────────────────────


class TestFindOptimalThreshold:
    def test_returns_dict_with_required_keys(self):
        from src.models.simulate_threshold import find_optimal_threshold

        y_true = np.array([0, 0, 1, 1])
        y_score = np.array([0.1, 0.2, 0.8, 0.9])
        result = find_optimal_threshold(y_true, y_score, target_recall=0.95)

        assert isinstance(result, dict)
        for key in ("threshold", "precision", "recall", "target_recall", "target_reached"):
            assert key in result, f"Clé manquante : {key}"

    def test_target_reached_when_recall_achievable(self):
        from src.models.simulate_threshold import find_optimal_threshold

        # Scores parfaits → recall 1.0 atteignable
        y_true = np.array([0, 0, 1, 1])
        y_score = np.array([0.1, 0.2, 0.8, 0.9])
        result = find_optimal_threshold(y_true, y_score, target_recall=0.95)

        assert result["target_reached"] is True
        assert result["recall"] >= 0.95

    def test_target_not_reached_when_recall_unachievable(self):
        from src.models.simulate_threshold import find_optimal_threshold

        # target_recall > 1.0 est mathématiquement impossible (recall ∈ [0,1])
        # Garantit que la condition n'est jamais satisfaite quel que soit le dataset
        y_true = np.array([0, 0, 1, 1])
        y_score = np.array([0.1, 0.2, 0.8, 0.9])
        result = find_optimal_threshold(y_true, y_score, target_recall=1.01)

        assert result["target_reached"] is False

    def test_threshold_in_valid_range(self):
        from src.models.simulate_threshold import find_optimal_threshold

        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_score = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        result = find_optimal_threshold(y_true, y_score, target_recall=0.90)

        assert 0.0 <= result["threshold"] <= 1.0

    def test_precision_in_valid_range(self):
        from src.models.simulate_threshold import find_optimal_threshold

        y_true = np.array([0, 0, 1, 1])
        y_score = np.array([0.1, 0.2, 0.8, 0.9])
        result = find_optimal_threshold(y_true, y_score, target_recall=0.90)

        assert 0.0 <= result["precision"] <= 1.0

    def test_target_recall_stored_in_result(self):
        from src.models.simulate_threshold import find_optimal_threshold

        y_true = np.array([0, 1])
        y_score = np.array([0.2, 0.8])
        result = find_optimal_threshold(y_true, y_score, target_recall=0.80)

        assert result["target_recall"] == 0.80

    def test_higher_threshold_lower_recall(self):
        from src.models.simulate_threshold import find_optimal_threshold

        y_true = np.array([0, 0, 1, 1, 1])
        y_score = np.array([0.2, 0.3, 0.6, 0.8, 0.9])

        r90 = find_optimal_threshold(y_true, y_score, target_recall=0.90)
        r50 = find_optimal_threshold(y_true, y_score, target_recall=0.50)

        # Un recall cible plus bas autorise un seuil plus haut (plus sélectif)
        assert r90["threshold"] <= r50["threshold"]


# ── validate_dataframe ────────────────────────────────────────────────────────


class TestValidateDataframe:
    def test_passes_on_valid_df(self):
        import pandas as pd

        from src.models.simulate_threshold import validate_dataframe

        df = pd.DataFrame({"label": [0, 1], "score": [0.2, 0.8]})
        validate_dataframe(df)  # ne doit pas lever

    def test_raises_on_missing_label(self):
        import pandas as pd

        from src.models.simulate_threshold import validate_dataframe

        df = pd.DataFrame({"score": [0.2, 0.8]})
        with pytest.raises(ValueError, match="label"):
            validate_dataframe(df)

    def test_raises_on_missing_score(self):
        import pandas as pd

        from src.models.simulate_threshold import validate_dataframe

        df = pd.DataFrame({"label": [0, 1]})
        with pytest.raises(ValueError, match="score"):
            validate_dataframe(df)

    def test_raises_on_nan_score(self):
        import pandas as pd

        from src.models.simulate_threshold import validate_dataframe

        df = pd.DataFrame({"label": [0, 1], "score": [float("nan"), 0.8]})
        with pytest.raises(ValueError):
            validate_dataframe(df)

    def test_raises_on_nan_label(self):
        import pandas as pd

        from src.models.simulate_threshold import validate_dataframe

        df = pd.DataFrame({"label": [float("nan"), 1], "score": [0.2, 0.8]})
        with pytest.raises(ValueError):
            validate_dataframe(df)


# ── load_from_csv ─────────────────────────────────────────────────────────────


class TestLoadFromCsv:
    def test_loads_valid_csv(self, tmp_path):
        import pandas as pd

        from src.models.simulate_threshold import load_from_csv

        csv_path = tmp_path / "preds.csv"
        pd.DataFrame({"label": [0, 1], "score": [0.2, 0.8]}).to_csv(csv_path, index=False)

        df = load_from_csv(str(csv_path))
        assert len(df) == 2
        assert "label" in df.columns
        assert "score" in df.columns

    def test_raises_on_missing_file(self, tmp_path):
        from src.models.simulate_threshold import load_from_csv

        with pytest.raises(FileNotFoundError):
            load_from_csv(str(tmp_path / "nonexistent.csv"))
