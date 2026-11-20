"""Pytest configuration and shared fixtures."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Make src/ importable the way the pipeline modules import each other
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture
def project_root():
    """Return project root directory."""
    return Path(__file__).parents[1]


@pytest.fixture
def sample_train_data():
    """Create sample training data for testing.

    Returns a small DataFrame mimicking train.csv structure (NO store metadata).
    """
    np.random.seed(42)
    n_stores = 5
    n_days = 30

    data = []
    for store in range(1, n_stores + 1):
        for day in range(n_days):
            date = pd.Timestamp("2015-01-01") + pd.Timedelta(days=day)
            is_open = 1 if date.dayofweek < 6 else 0

            data.append(
                {
                    "Store": store,
                    "DayOfWeek": date.dayofweek + 1,
                    "Date": date,
                    "Sales": np.random.randint(3000, 8000) if is_open else 0,
                    "Customers": np.random.randint(400, 1200) if is_open else 0,
                    "Open": is_open,
                    "Promo": np.random.choice([0, 1]) if is_open else 0,
                    "StateHoliday": "0",
                    "SchoolHoliday": 0,
                }
            )

    return pd.DataFrame(data)


@pytest.fixture
def sample_store_data():
    """Create sample store data for testing.

    Returns a small DataFrame mimicking store.csv structure.
    """
    stores = []
    for store_id in range(1, 6):
        stores.append(
            {
                "Store": store_id,
                "StoreType": np.random.choice(["a", "b", "c", "d"]),
                "Assortment": np.random.choice(["a", "b", "c"]),
                "CompetitionDistance": np.random.randint(100, 5000),
                "CompetitionOpenSinceMonth": np.random.randint(1, 13),
                "CompetitionOpenSinceYear": np.random.randint(2010, 2015),
                "Promo2": np.random.choice([0, 1]),
                "Promo2SinceWeek": np.random.randint(1, 53),
                "Promo2SinceYear": 2014,
                "PromoInterval": "Feb,May,Aug,Nov",
            }
        )

    return pd.DataFrame(stores)


@pytest.fixture
def sample_features_data(sample_train_data, sample_store_data):
    """Sample data run through the real cleaning and feature pipeline."""
    from data.make_dataset import basic_cleaning, merge_store_info
    from features.build_features import build_all_features

    return build_all_features(
        basic_cleaning(merge_store_info(sample_train_data, sample_store_data))
    )


@pytest.fixture
def temp_output_dir(tmp_path):
    """Create temporary output directory for tests."""
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    return output_dir


@pytest.fixture
def mock_model_version():
    """Create a mock MLflow ModelVersion object.

    Returns
    -------
    Mock
        Mock ModelVersion with common attributes
    """
    from unittest.mock import Mock

    mv = Mock()
    mv.name = "rossmann-model"
    mv.version = "1"
    mv.current_stage = "Production"
    mv.description = "Test LightGBM model"
    mv.run_id = "test-run-id-123"
    mv.status = "READY"
    mv.creation_timestamp = 1609459200000  # 2021-01-01
    mv.last_updated_timestamp = 1609459200000
    return mv


@pytest.fixture
def mock_registered_model():
    """Create a mock MLflow RegisteredModel object.

    Returns
    -------
    Mock
        Mock RegisteredModel with common attributes
    """
    from unittest.mock import Mock

    rm = Mock()
    rm.name = "rossmann-model"
    rm.description = "LightGBM model for Rossmann sales forecasting"
    rm.creation_timestamp = 1609459200000
    rm.last_updated_timestamp = 1609459200000
    return rm
