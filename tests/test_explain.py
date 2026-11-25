"""Tests for SHAP values (src/models/explain.py)."""

import numpy as np

from models import train
from models.explain import compute_shap_values

PARAMS = {
    "objective": "regression",
    "num_leaves": 7,
    "learning_rate": 0.1,
    "num_boost_round": 20,
    "seed": 7075,
    "num_threads": 1,
    "verbose": -1,
}


def _model_and_rows(sample_features_data):
    X, y = train.prepare_features(sample_features_data)
    return train.train_model(X, y, PARAMS), X.head(5)


def test_one_column_per_feature_plus_base_value(sample_features_data):
    model, X = _model_and_rows(sample_features_data)
    shap_df = compute_shap_values(model, X)
    assert shap_df.columns.tolist() == X.columns.tolist() + ["base_value"]
    assert shap_df.index.equals(X.index)


def test_contributions_sum_to_prediction(sample_features_data):
    model, X = _model_and_rows(sample_features_data)
    shap_df = compute_shap_values(model, X)
    np.testing.assert_allclose(shap_df.sum(axis=1).values, model.predict(X), rtol=1e-6)
