"""
Unit tests for src/monitoring/generate_predictions.py
"""

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import torch

# ── PredictionDataset ─────────────────────────────────────────────────────────


class TestPredictionDataset:
    def test_len(self, tmp_path):
        from src.monitoring.generate_predictions import PredictionDataset

        imgs = []
        for i in range(5):
            p = tmp_path / f"img_{i}.png"
            from PIL import Image

            Image.fromarray(np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)).save(p)
            imgs.append(str(p))

        df = pd.DataFrame({"image_path": imgs, "label": [0, 1, 0, 1, 0]})
        ds = PredictionDataset(df)
        assert len(ds) == 5

    def test_getitem_returns_tensor_label_id(self, tmp_path):
        from PIL import Image

        from src.monitoring.generate_predictions import PredictionDataset

        p = tmp_path / "img.png"
        Image.fromarray(np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)).save(p)

        df = pd.DataFrame({"image_path": [str(p)], "label": [1]})
        ds = PredictionDataset(df)
        tensor, label, sample_id = ds[0]

        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (3, 224, 224)
        assert label == 1
        assert sample_id == "img"

    def test_missing_image_returns_zeros(self, tmp_path):
        from src.monitoring.generate_predictions import PredictionDataset

        df = pd.DataFrame(
            {
                "image_path": [str(tmp_path / "nonexistent.png")],
                "label": [0],
            }
        )
        ds = PredictionDataset(df)
        tensor, label, _ = ds[0]

        assert tensor.shape == (3, 224, 224)
        assert torch.all(tensor == 0), "Image illisible doit retourner un tenseur de zéros"

    def test_no_label_column_uses_minus_one(self, tmp_path):
        from PIL import Image

        from src.monitoring.generate_predictions import PredictionDataset

        p = tmp_path / "img.png"
        Image.fromarray(np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)).save(p)

        df = pd.DataFrame({"image_path": [str(p)]})
        ds = PredictionDataset(df)
        _, label, _ = ds[0]
        assert label == -1


# ── run_inference ─────────────────────────────────────────────────────────────


class TestRunInference:
    def _make_df(self, tmp_path, n=6):
        from PIL import Image

        rows = []
        for i in range(n):
            p = tmp_path / f"img_{i}.png"
            Image.fromarray(np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)).save(p)
            rows.append({"image_path": str(p), "label": i % 2})
        return pd.DataFrame(rows)

    def _make_model(self):
        model = MagicMock()
        model.eval = MagicMock(return_value=model)
        # softmax([0, 1]) → prob fraude ≈ 0.73
        model.return_value = torch.tensor([[0.0, 1.0]] * 32)
        return model

    def test_returns_list_of_dicts(self, tmp_path):
        from src.monitoring.generate_predictions import run_inference

        df = self._make_df(tmp_path, n=4)
        model = self._make_model()
        records = run_inference(model, df, threshold=0.5)

        assert isinstance(records, list)
        assert len(records) == 4
        assert all(isinstance(r, dict) for r in records)

    def test_record_has_required_keys(self, tmp_path):
        from src.monitoring.generate_predictions import run_inference

        df = self._make_df(tmp_path, n=2)
        model = self._make_model()
        records = run_inference(model, df, threshold=0.5)

        for key in ("id", "label", "score", "prediction", "timestamp"):
            assert key in records[0], f"Clé manquante : {key}"

    def test_score_in_valid_range(self, tmp_path):
        from src.monitoring.generate_predictions import run_inference

        df = self._make_df(tmp_path, n=4)
        model = self._make_model()
        records = run_inference(model, df, threshold=0.5)

        for r in records:
            assert 0.0 <= r["score"] <= 1.0

    def test_prediction_is_binary(self, tmp_path):
        from src.monitoring.generate_predictions import run_inference

        df = self._make_df(tmp_path, n=4)
        model = self._make_model()
        records = run_inference(model, df, threshold=0.5)

        for r in records:
            assert r["prediction"] in (0, 1)

    def test_max_samples_limits_output(self, tmp_path):
        from src.monitoring.generate_predictions import run_inference

        df = self._make_df(tmp_path, n=10)
        model = self._make_model()
        records = run_inference(model, df, threshold=0.5, max_samples=4)

        assert len(records) == 4

    def test_threshold_determines_prediction(self, tmp_path):
        from src.monitoring.generate_predictions import run_inference

        df = self._make_df(tmp_path, n=2)
        model = self._make_model()

        # score ≈ 0.73 (softmax de [0,1])
        records_low = run_inference(model, df, threshold=0.5)
        records_high = run_inference(model, df, threshold=0.99)

        assert all(r["prediction"] == 1 for r in records_low)
        assert all(r["prediction"] == 0 for r in records_high)


# ── push_to_s3 ────────────────────────────────────────────────────────────────


class TestPushToS3:
    def test_uploads_one_file_per_record(self):
        from src.monitoring.generate_predictions import push_to_s3

        records = [
            {"id": "img_001", "score": 0.8, "prediction": 1, "label": 1, "timestamp": "t"},
            {"id": "img_002", "score": 0.2, "prediction": 0, "label": 0, "timestamp": "t"},
        ]

        mock_s3 = MagicMock()
        with (
            patch("src.monitoring.generate_predictions.boto3.client", return_value=mock_s3),
            patch("src.monitoring.generate_predictions.S3_BUCKET", "test-bucket"),
        ):
            n = push_to_s3(records, prefix="predictions/test")

        assert n == 2
        assert mock_s3.put_object.call_count == 2

    def test_key_contains_prefix_and_id(self):
        from src.monitoring.generate_predictions import push_to_s3

        records = [{"id": "img_abc", "score": 0.5, "prediction": 0, "label": 0, "timestamp": "t"}]

        mock_s3 = MagicMock()
        with (
            patch("src.monitoring.generate_predictions.boto3.client", return_value=mock_s3),
            patch("src.monitoring.generate_predictions.S3_BUCKET", "test-bucket"),
        ):
            push_to_s3(records, prefix="predictions/train_ref")

        call_kwargs = mock_s3.put_object.call_args[1]
        assert call_kwargs["Key"] == "predictions/train_ref/img_abc.json"

    def test_raises_when_bucket_not_set(self):
        from src.monitoring.generate_predictions import push_to_s3

        with patch("src.monitoring.generate_predictions.S3_BUCKET", None):
            with pytest.raises(RuntimeError, match="S3_BUCKET"):
                push_to_s3([{"id": "x"}], prefix="p")

    def test_body_is_valid_json(self):
        from src.monitoring.generate_predictions import push_to_s3

        records = [{"id": "img_001", "score": 0.9, "prediction": 1, "label": 1, "timestamp": "t"}]

        mock_s3 = MagicMock()
        with (
            patch("src.monitoring.generate_predictions.boto3.client", return_value=mock_s3),
            patch("src.monitoring.generate_predictions.S3_BUCKET", "test-bucket"),
        ):
            push_to_s3(records, prefix="predictions/test")

        body = mock_s3.put_object.call_args[1]["Body"]
        parsed = json.loads(body.decode("utf-8"))
        assert parsed["id"] == "img_001"
