"""Tests for data validation (src/data/validate_data.py and src/data/expectations.py).

The suites run against the real committed sample data (data/raw/), so a rule
that doesn't fit the data fails here, not in a student's pipeline run. Each
test uses its own Great Expectations folder under a temporary directory.
"""

import pandas as pd
import pytest

from data import expectations
from data import validate_data as vd
from data.make_dataset import basic_cleaning, merge_store_info
from features.build_features import build_all_features, get_model_features


@pytest.fixture(scope="module")
def context(tmp_path_factory):
    return vd.get_context(tmp_path_factory.mktemp("gx_project"))


@pytest.fixture(scope="module")
def raw_train():
    return vd.read_raw_train()


@pytest.fixture(scope="module")
def raw_store():
    return pd.read_csv(vd.RAW_STORE_PATH)


@pytest.fixture(scope="module")
def features(raw_train, raw_store):
    return build_all_features(basic_cleaning(merge_store_info(raw_train, raw_store)))


def _failed(result) -> set[tuple[str, str | None]]:
    """The (expectation type, column) pairs that failed."""
    return {
        (r.expectation_config.type, r.expectation_config.kwargs.get("column"))
        for r in result.results
        if not r.success
    }


class TestSuitesPassOnSampleData:
    def test_raw_store(self, context, raw_store):
        assert vd.run_checkpoint(context, raw_store, expectations.raw_store_suite()).success

    def test_raw_train(self, context, raw_train, raw_store):
        suite = expectations.raw_train_suite(raw_store["Store"].tolist())
        assert vd.run_checkpoint(context, raw_train, suite).success

    def test_features(self, context, features):
        assert vd.run_checkpoint(context, features, expectations.features_suite()).success


class TestSuitesCatchBadData:
    def test_negative_sales_and_unknown_store(self, context, raw_train, raw_store):
        bad = raw_train.copy()
        bad.loc[:2, "Sales"] = -100
        bad.loc[5, "Store"] = 9999
        suite = expectations.raw_train_suite(raw_store["Store"].tolist())

        result = vd.run_checkpoint(context, bad, suite)

        assert not result.success
        assert _failed(result) == {
            ("expect_column_values_to_be_between", "Sales"),
            ("expect_column_values_to_be_in_set", "Store"),
        }

    def test_bad_date_format(self, context, raw_train, raw_store):
        bad = raw_train.copy()
        bad.loc[0, "Date"] = "31/07/2015"
        suite = expectations.raw_train_suite(raw_store["Store"].tolist())

        result = vd.run_checkpoint(context, bad, suite)

        assert _failed(result) == {("expect_column_values_to_match_strftime_format", "Date")}

    def test_duplicate_store(self, context, raw_store):
        bad = pd.concat([raw_store, raw_store.iloc[[0]]], ignore_index=True)

        result = vd.run_checkpoint(context, bad, expectations.raw_store_suite())

        assert ("expect_column_values_to_be_unique", "Store") in _failed(result)
        assert ("expect_table_row_count_to_equal", None) in _failed(result)

    def test_missing_model_feature(self, context, features):
        feature = get_model_features()[0]
        bad = features.drop(columns=[feature])

        result = vd.run_checkpoint(context, bad, expectations.features_suite())

        assert ("expect_column_to_exist", feature) in _failed(result)

    def test_blank_feature_values(self, context, features):
        bad = features.copy()
        bad.loc[bad.index[:10], "CompetitionDistance"] = None

        result = vd.run_checkpoint(context, bad, expectations.features_suite())

        assert _failed(result) == {("expect_column_values_to_not_be_null", "CompetitionDistance")}


class TestFeaturesSuite:
    def test_checks_every_model_feature(self):
        suite = expectations.features_suite()
        checked = {
            e.column for e in suite.expectations if e.expectation_type == "expect_column_to_exist"
        }
        assert set(get_model_features()) <= checked

    def test_every_rule_has_a_note(self, raw_store):
        suites = [
            expectations.raw_train_suite(raw_store["Store"].tolist()),
            expectations.raw_store_suite(),
            expectations.features_suite(),
        ]
        for suite in suites:
            for expectation in suite.expectations:
                assert expectation.notes, expectation.expectation_type


class TestReporting:
    def test_describe_failures_explains_the_rule(self, context, raw_store):
        bad = raw_store.copy()
        bad.loc[0, "StoreType"] = "z"

        result = vd.run_checkpoint(context, bad, expectations.raw_store_suite())
        lines = vd.describe_failures(result)

        assert len(lines) == 1
        assert "StoreType: 1 row failed, for example ['z']" in lines[0]
        assert "Rossmann has four store formats" in lines[0]

    def test_data_docs_are_built(self, context, raw_store):
        vd.run_checkpoint(context, raw_store, expectations.raw_store_suite())
        docs = vd.DATA_DOCS_PATH.relative_to("gx")
        assert (context.root_directory / docs).exists()


@pytest.mark.integration
class TestMain:
    @pytest.fixture(autouse=True)
    def temp_project(self, tmp_path, monkeypatch):
        """Keep main()'s Great Expectations folder out of the repository."""
        real_get_context = vd.get_context
        monkeypatch.setattr(vd, "get_context", lambda: real_get_context(tmp_path))

    def test_exit_code_0_when_data_passes(self):
        assert vd.main(["--stage", "raw"]) == 0

    def test_exit_code_1_when_data_fails(self, tmp_path, raw_store, monkeypatch):
        bad_store = tmp_path / "store.csv"
        raw_store.assign(Promo2=7).to_csv(bad_store, index=False)
        monkeypatch.setattr(vd, "RAW_STORE_PATH", bad_store)

        assert vd.main(["--stage", "raw"]) == 1
