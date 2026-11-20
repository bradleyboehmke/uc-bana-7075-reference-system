"""Tests for feature engineering (src/features/build_features.py)."""

import sys
from pathlib import Path

import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.make_dataset import basic_cleaning, merge_store_info
from features.build_features import (
    add_calendar_features,
    add_promo_features,
    build_all_features,
    get_categorical_features,
    get_model_features,
)


def _promo_frame(dates, promo2, interval):
    return pd.DataFrame(
        {
            "Date": pd.to_datetime(dates),
            "Promo2": promo2,
            "PromoInterval": interval,
        }
    )


class TestModelFeatureList:
    """The feature list in config/params.yaml."""

    def test_has_twelve_features(self):
        assert len(get_model_features()) == 12

    def test_no_sales_history_features(self):
        """Features must be known in advance; none may be derived from Sales or Customers."""
        for feature in get_model_features():
            assert "Sales" not in feature
            assert "Customers" not in feature

    def test_categorical_features_are_model_features(self):
        assert set(get_categorical_features()) <= set(get_model_features())


class TestCalendarFeatures:
    def test_calendar_columns_created(self, sample_train_data):
        df = sample_train_data.copy()
        df["Date"] = pd.to_datetime(df["Date"])
        result = add_calendar_features(df)
        for col in ["Year", "Month", "Week"]:
            assert col in result.columns

    def test_values_match_date(self):
        df = pd.DataFrame({"Date": pd.to_datetime(["2015-01-01", "2015-12-28"])})
        result = add_calendar_features(df)
        assert result["Year"].tolist() == [2015, 2015]
        assert result["Month"].tolist() == [1, 12]
        assert result["Week"].tolist() == [1, 53]


class TestPromoFeatures:
    def test_active_in_listed_month(self):
        df = _promo_frame(["2015-02-10", "2015-03-10"], [1, 1], ["Feb,May,Aug,Nov"] * 2)
        assert add_promo_features(df)["Promo2Active"].tolist() == [1, 0]

    def test_inactive_when_store_not_in_promo2(self):
        df = _promo_frame(["2015-02-10"], [0], ["Feb,May,Aug,Nov"])
        assert add_promo_features(df)["Promo2Active"].tolist() == [0]

    def test_september_spelling(self):
        """store.csv spells September 'Sept'."""
        df = _promo_frame(["2015-09-15"], [1], ["Mar,Jun,Sept,Dec"])
        assert add_promo_features(df)["Promo2Active"].tolist() == [1]

    def test_empty_interval(self):
        df = _promo_frame(["2015-02-10"], [1], [""])
        assert add_promo_features(df)["Promo2Active"].tolist() == [0]


class TestBuildAllFeatures:
    def test_output_contains_every_model_feature(self, sample_train_data, sample_store_data):
        clean = basic_cleaning(merge_store_info(sample_train_data, sample_store_data))
        result = build_all_features(clean)
        missing = set(get_model_features()) - set(result.columns)
        assert not missing

    def test_no_nulls_in_model_features(self, sample_train_data, sample_store_data):
        clean = basic_cleaning(merge_store_info(sample_train_data, sample_store_data))
        result = build_all_features(clean)
        assert result[get_model_features()].isna().sum().sum() == 0

    def test_row_count_unchanged(self, sample_train_data, sample_store_data):
        clean = basic_cleaning(merge_store_info(sample_train_data, sample_store_data))
        assert len(build_all_features(clean)) == len(clean)

    def test_single_row_matches_batch(self, sample_train_data, sample_store_data):
        """Features for one row don't depend on other rows (no training/serving skew)."""
        clean = basic_cleaning(merge_store_info(sample_train_data, sample_store_data))
        batch = build_all_features(clean)
        one = build_all_features(clean.iloc[[10]])
        features = get_model_features()
        row = batch[
            (batch["Store"] == one["Store"].iloc[0]) & (batch["Date"] == one["Date"].iloc[0])
        ]
        pd.testing.assert_frame_equal(
            row[features].reset_index(drop=True).astype(str),
            one[features].reset_index(drop=True).astype(str),
        )
