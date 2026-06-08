"""
Test suite for mlops-airflow-pipeline.
Tests each task independently without requiring Airflow.
Run: pytest tests/ -v --cov=src
"""

import pytest
import os
import sys
import json
import shutil
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.tasks.ingest   import ingest_data
from src.tasks.features import engineer_features
from src.tasks.train    import train_model
from src.tasks.validate import validate_model

TEST_DIR = "test_artifacts"


@pytest.fixture(autouse=True)
def setup_teardown():
    os.makedirs(TEST_DIR, exist_ok=True)
    os.makedirs(f"{TEST_DIR}/processed", exist_ok=True)
    yield
    shutil.rmtree(TEST_DIR, ignore_errors=True)


class TestIngest:
    def test_ingest_synthetic(self):
        meta = ingest_data(output_dir=f"{TEST_DIR}/processed")
        assert meta["n_rows"] > 0
        assert meta["source"] == "synthetic"

    def test_ingest_output_exists(self):
        meta = ingest_data(output_dir=f"{TEST_DIR}/processed")
        assert os.path.exists(meta["output_path"])

    def test_ingest_positive_rate(self):
        meta = ingest_data(output_dir=f"{TEST_DIR}/processed")
        assert 0 < meta["positive_rate"] < 1

    def test_ingest_returns_metadata(self):
        meta = ingest_data(output_dir=f"{TEST_DIR}/processed")
        for key in ["n_rows","n_cols","positive_rate","source","output_path"]:
            assert key in meta


class TestFeatures:
    def test_engineer_features(self):
        ingest_meta = ingest_data(output_dir=f"{TEST_DIR}/processed")
        feat_meta = engineer_features(
            input_path=ingest_meta["output_path"],
            output_dir=f"{TEST_DIR}/processed"
        )
        assert feat_meta["n_features"] > 0

    def test_features_output_exists(self):
        ingest_meta = ingest_data(output_dir=f"{TEST_DIR}/processed")
        feat_meta = engineer_features(
            input_path=ingest_meta["output_path"],
            output_dir=f"{TEST_DIR}/processed"
        )
        assert os.path.exists(feat_meta["output_path"])
        assert os.path.exists(feat_meta["encoders_path"])

    def test_encoders_saved(self):
        ingest_meta = ingest_data(output_dir=f"{TEST_DIR}/processed")
        feat_meta = engineer_features(
            input_path=ingest_meta["output_path"],
            output_dir=f"{TEST_DIR}/processed"
        )
        encoders = json.load(open(feat_meta["encoders_path"]))
        assert len(encoders) > 0


class TestTrain:
    def test_train_model(self):
        ingest_meta = ingest_data(output_dir=f"{TEST_DIR}/processed")
        feat_meta = engineer_features(
            input_path=ingest_meta["output_path"],
            output_dir=f"{TEST_DIR}/processed"
        )
        train_meta = train_model(
            input_path=feat_meta["output_path"],
            output_dir=f"{TEST_DIR}/artifacts"
        )
        assert train_meta["auc"] > 0.4

    def test_model_saved(self):
        ingest_meta = ingest_data(output_dir=f"{TEST_DIR}/processed")
        feat_meta = engineer_features(
            input_path=ingest_meta["output_path"],
            output_dir=f"{TEST_DIR}/processed"
        )
        train_meta = train_model(
            input_path=feat_meta["output_path"],
            output_dir=f"{TEST_DIR}/artifacts"
        )
        assert os.path.exists(train_meta["model_path"])

    def test_cv_score(self):
        ingest_meta = ingest_data(output_dir=f"{TEST_DIR}/processed")
        feat_meta = engineer_features(
            input_path=ingest_meta["output_path"],
            output_dir=f"{TEST_DIR}/processed"
        )
        train_meta = train_model(
            input_path=feat_meta["output_path"],
            output_dir=f"{TEST_DIR}/artifacts"
        )
        assert train_meta["cv_auc_mean"] > 0.4
        assert train_meta["cv_auc_std"] >= 0


class TestValidate:
    def _get_train_meta(self):
        ingest_meta = ingest_data(output_dir=f"{TEST_DIR}/processed")
        feat_meta = engineer_features(
            input_path=ingest_meta["output_path"],
            output_dir=f"{TEST_DIR}/processed"
        )
        return train_model(
            input_path=feat_meta["output_path"],
            output_dir=f"{TEST_DIR}/artifacts"
        )

    def test_validate_passes(self):
        train_meta = self._get_train_meta()
        meta = validate_model(train_meta=train_meta, auc_threshold=0.4)
        assert meta["passed"] is True

    def test_validate_fails(self):
        train_meta = self._get_train_meta()
        with pytest.raises(ValueError):
            validate_model(train_meta=train_meta, auc_threshold=0.99)

    def test_validate_returns_meta(self):
        train_meta = self._get_train_meta()
        meta = validate_model(train_meta=train_meta, auc_threshold=0.4)
        for key in ["auc","cv_auc","threshold","passed"]:
            assert key in meta
