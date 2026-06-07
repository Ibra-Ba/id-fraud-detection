"""
Unit tests for src/monitoring/build_reference_csv.py
"""

import json
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

# ── build_dataframe ───────────────────────────────────────────────────────────


class TestBuildDataframe:
    def _records(self):
        return [
            {"id": "img_001", "label": 0, "score": 0.1, "prediction": 0, "timestamp": "t"},
            {"id": "img_002", "label": 1, "score": 0.9, "prediction": 1, "timestamp": "t"},
            {"id": "img_003", "label": 0, "score": 0.2, "prediction": 0, "timestamp": "t"},
        ]

    def test_returns_dataframe(self):
        from src.monitoring.build_reference_csv import build_dataframe

        df = build_dataframe(self._records())
        assert isinstance(df, pd.DataFrame)

    def test_correct_row_count(self):
        from src.monitoring.build_reference_csv import build_dataframe

        df = build_dataframe(self._records())
        assert len(df) == 3

    def test_required_columns_present(self):
        from src.monitoring.build_reference_csv import build_dataframe

        df = build_dataframe(self._records())
        for col in ("label", "score", "prediction"):
            assert col in df.columns, f"Colonne manquante : {col}"

    def test_score_dtype_float(self):
        from src.monitoring.build_reference_csv import build_dataframe

        df = build_dataframe(self._records())
        assert df["score"].dtype == float

    def test_prediction_dtype_int(self):
        from src.monitoring.build_reference_csv import build_dataframe

        df = build_dataframe(self._records())
        assert df["prediction"].dtype in (int, "int64", "int32")

    def test_label_dtype_int(self):
        from src.monitoring.build_reference_csv import build_dataframe

        df = build_dataframe(self._records())
        assert df["label"].dtype in (int, "int64", "int32")

    def test_drops_rows_with_none_label(self):
        from src.monitoring.build_reference_csv import build_dataframe

        records = self._records()
        records.append(
            {"id": "img_004", "label": None, "score": 0.5, "prediction": 0, "timestamp": "t"}
        )
        df = build_dataframe(records)
        assert len(df) == 3

    def test_drops_duplicate_ids(self):
        from src.monitoring.build_reference_csv import build_dataframe

        records = self._records()
        records.append(
            {"id": "img_001", "label": 1, "score": 0.8, "prediction": 1, "timestamp": "t"}
        )
        df = build_dataframe(records)
        assert len(df) == 3

    def test_id_is_first_column_when_present(self):
        from src.monitoring.build_reference_csv import build_dataframe

        df = build_dataframe(self._records())
        assert df.columns[0] == "id"

    def test_empty_records_returns_empty_df(self):
        import pandas as pd

        from src.monitoring.build_reference_csv import build_dataframe

        df = build_dataframe([])
        assert len(df) == 0
        assert isinstance(df, pd.DataFrame)


# ── load_records_from_s3 ──────────────────────────────────────────────────────


class TestLoadRecordsFromS3:
    def _mock_s3_with_records(self, records):
        """Construit un mock boto3 qui retourne les records comme fichiers JSON S3."""
        mock_s3 = MagicMock()

        # Simule list_objects_v2 via paginator
        mock_paginator = MagicMock()
        mock_s3.get_paginator.return_value = mock_paginator

        pages = [{"Contents": [{"Key": f"prefix/{r['id']}.json"} for r in records]}]
        mock_paginator.paginate.return_value = iter(pages)

        # Simule get_object pour chaque clé
        def fake_get_object(Bucket, Key):
            record_id = Key.split("/")[-1].replace(".json", "")
            record = next(r for r in records if r["id"] == record_id)
            body = MagicMock()
            body.read.return_value = json.dumps(record).encode()
            return {"Body": body}

        mock_s3.get_object.side_effect = fake_get_object
        return mock_s3

    def test_returns_list(self):
        from src.monitoring.build_reference_csv import load_records_from_s3

        records = [
            {"id": "img_001", "label": 0, "score": 0.1, "prediction": 0, "timestamp": "t"},
        ]
        mock_s3 = self._mock_s3_with_records(records)

        with (
            patch("src.monitoring.build_reference_csv.boto3.client", return_value=mock_s3),
            patch("src.monitoring.build_reference_csv.S3_BUCKET", "test-bucket"),
        ):
            result = load_records_from_s3("prefix")

        assert isinstance(result, list)
        assert len(result) == 1

    def test_raises_when_no_files(self):
        from src.monitoring.build_reference_csv import load_records_from_s3

        mock_s3 = MagicMock()
        mock_paginator = MagicMock()
        mock_s3.get_paginator.return_value = mock_paginator
        mock_paginator.paginate.return_value = iter([{"Contents": []}])

        with (
            patch("src.monitoring.build_reference_csv.boto3.client", return_value=mock_s3),
            patch("src.monitoring.build_reference_csv.S3_BUCKET", "test-bucket"),
        ):
            with pytest.raises(ValueError, match="Aucun fichier JSON"):
                load_records_from_s3("empty/prefix")

    def test_raises_when_bucket_not_set(self):
        from src.monitoring.build_reference_csv import load_records_from_s3

        with patch("src.monitoring.build_reference_csv.S3_BUCKET", None):
            with pytest.raises(RuntimeError, match="S3_BUCKET"):
                load_records_from_s3("prefix")

    def test_ignores_non_json_files(self):
        from src.monitoring.build_reference_csv import load_records_from_s3

        mock_s3 = MagicMock()
        mock_paginator = MagicMock()
        mock_s3.get_paginator.return_value = mock_paginator

        # Un fichier .csv et un fichier .json
        pages = [
            {
                "Contents": [
                    {"Key": "prefix/manifest.csv"},
                    {"Key": "prefix/img_001.json"},
                ]
            }
        ]
        mock_paginator.paginate.return_value = iter(pages)

        record = {"id": "img_001", "label": 0, "score": 0.1, "prediction": 0, "timestamp": "t"}
        body = MagicMock()
        body.read.return_value = json.dumps(record).encode()
        mock_s3.get_object.return_value = {"Body": body}

        with (
            patch("src.monitoring.build_reference_csv.boto3.client", return_value=mock_s3),
            patch("src.monitoring.build_reference_csv.S3_BUCKET", "test-bucket"),
        ):
            result = load_records_from_s3("prefix")

        # Seul le .json doit être chargé
        assert len(result) == 1
        assert mock_s3.get_object.call_count == 1


# ── save_local ────────────────────────────────────────────────────────────────


class TestSaveLocal:
    def test_creates_file(self, tmp_path):
        from src.monitoring.build_reference_csv import save_local

        df = pd.DataFrame({"label": [0, 1], "score": [0.2, 0.8], "prediction": [0, 1]})
        output_path = str(tmp_path / "out" / "train_with_preds.csv")

        path = save_local(df, output_path)

        assert path.exists()

    def test_file_is_readable_csv(self, tmp_path):
        from src.monitoring.build_reference_csv import save_local

        df = pd.DataFrame({"label": [0, 1], "score": [0.2, 0.8], "prediction": [0, 1]})
        output_path = str(tmp_path / "train_with_preds.csv")

        path = save_local(df, output_path)
        loaded = pd.read_csv(path)

        assert len(loaded) == 2
        assert list(loaded.columns) == ["label", "score", "prediction"]

    def test_creates_parent_dirs(self, tmp_path):
        from src.monitoring.build_reference_csv import save_local

        df = pd.DataFrame({"label": [0], "score": [0.5], "prediction": [0]})
        nested = str(tmp_path / "a" / "b" / "c" / "out.csv")

        path = save_local(df, nested)
        assert path.exists()
