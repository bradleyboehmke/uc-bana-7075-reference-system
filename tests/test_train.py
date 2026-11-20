"""Tests for model training (src/models/train.py)."""

from unittest.mock import patch

import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd
import pytest

from features.build_features import get_model_features
from models import train
from utils.io import load_params

SMALL_PARAMS = {
    "objective": "regression",
    "num_leaves": 7,
    "learning_rate": 0.1,
    "num_boost_round": 20,
    "seed": 7075,
    "num_threads": 1,
    "verbose": -1,
}


class TestSplitHoldout:
    def test_holdout_is_last_n_days(self, sample_features_data):
        train_df, holdout_df = train.split_holdout(sample_features_data, holdout_days=7)
        assert holdout_df["Date"].nunique() == 7
        assert holdout_df["Date"].max() == sample_features_data["Date"].max()

    def test_no_overlap_and_nothing_lost(self, sample_features_data):
        train_df, holdout_df = train.split_holdout(sample_features_data, holdout_days=7)
        assert train_df["Date"].max() < holdout_df["Date"].min()
        assert len(train_df) + len(holdout_df) == len(sample_features_data)


class TestPrepareFeatures:
    def test_uses_configured_features_only(self, sample_features_data):
        X, y = train.prepare_features(sample_features_data)
        assert X.columns.tolist() == get_model_features()

    def test_keeps_open_days_only(self, sample_features_data):
        X, y = train.prepare_features(sample_features_data)
        assert len(X) == (sample_features_data["Open"] == 1).sum()
        assert len(y) == len(X)

    def test_categoricals_have_category_dtype(self, sample_features_data):
        X, _ = train.prepare_features(sample_features_data)
        for col in ["StoreType", "Assortment", "StateHoliday"]:
            assert X[col].dtype.name == "category"


class TestTrainModel:
    def test_returns_booster_with_requested_rounds(self, sample_features_data):
        X, y = train.prepare_features(sample_features_data)
        model = train.train_model(X, y, SMALL_PARAMS)
        assert isinstance(model, lgb.Booster)
        assert model.num_trees() == 20

    def test_does_not_mutate_params(self, sample_features_data):
        X, y = train.prepare_features(sample_features_data)
        params = dict(SMALL_PARAMS)
        train.train_model(X, y, params)
        assert params == SMALL_PARAMS

    def test_deterministic(self, sample_features_data):
        X, y = train.prepare_features(sample_features_data)
        p1 = train.train_model(X, y, SMALL_PARAMS).predict(X)
        p2 = train.train_model(X, y, SMALL_PARAMS).predict(X)
        np.testing.assert_allclose(p1, p2)

    def test_config_has_fixed_hyperparameters(self):
        params = load_params()["model"]["lightgbm"]
        assert {"objective", "learning_rate", "num_leaves", "num_boost_round", "seed"} <= set(
            params
        )


class TestEvaluate:
    def test_returns_metrics(self, sample_features_data):
        X, y = train.prepare_features(sample_features_data)
        model = train.train_model(X, y, SMALL_PARAMS)
        metrics = train.evaluate(model, X, y)
        assert set(metrics) == {"rmspe", "rmse", "mae"}
        assert all(v >= 0 for v in metrics.values())


class TestMain:
    @pytest.mark.integration
    def test_trains_logs_and_registers(self, sample_features_data, tmp_path):
        data_path = tmp_path / "features.parquet"
        sample_features_data.to_parquet(data_path)
        tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"

        def fake_setup_mlflow():
            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment("test")

        config = load_params()
        config["model"]["lightgbm"] = dict(SMALL_PARAMS)
        config["training"]["holdout_days"] = 7

        with (
            patch.object(train, "setup_mlflow", fake_setup_mlflow),
            patch.object(train, "load_params", return_value=config),
        ):
            result = train.main(data_path=str(data_path))

        assert result["version"] == "1"
        assert result["metrics"]["rmspe"] > 0

        run = mlflow.get_run(result["run_id"])
        assert run.data.params["num_leaves"] == "7"
        assert "rmspe" in run.data.metrics

        loaded = mlflow.pyfunc.load_model(f"models:/{config['mlflow']['model_name']}/1")
        X, _ = train.prepare_features(sample_features_data)
        assert len(loaded.predict(pd.DataFrame(X))) == len(X)
