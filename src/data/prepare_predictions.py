"""Prepare new data for model predictions.

This module provides a unified interface for preparing raw prediction data through
the complete data processing pipeline:
1. Load and merge store metadata
2. Clean data
3. Engineer features
4. Extract model-ready features

This ensures consistency between training and inference data processing: the API
builds features with the same functions training uses.
"""

import sys
from pathlib import Path
from typing import Union

import pandas as pd

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from data.make_dataset import basic_cleaning, merge_store_info
from features.build_features import build_all_features, get_categorical_features, get_model_features
from utils.io import PROJECT_ROOT, read_csv
from utils.log import get_logger

logger = get_logger(__name__)

STORE_PATH = PROJECT_ROOT / "data" / "raw" / "store.csv"


def get_known_stores(store_metadata: pd.DataFrame | None = None) -> set[int]:
    """Return the store IDs the system has data for (the stores in store.csv).

    The model only knows these stores, so requests for any other store ID are rejected.

    Parameters
    ----------
    store_metadata : pd.DataFrame, optional
        Store metadata. If None, loaded from data/raw/store.csv

    Returns
    -------
    set of int
        Known store IDs
    """
    if store_metadata is None:
        store_metadata = read_csv(STORE_PATH)
    return set(store_metadata["Store"].astype(int))


def build_prediction_features(
    raw_data: pd.DataFrame,
    store_metadata: pd.DataFrame | None = None,
    store_path: Union[str, Path] = STORE_PATH,
) -> pd.DataFrame:
    """Run raw request rows through the same cleaning and feature steps as training.

    Parameters
    ----------
    raw_data : pd.DataFrame
        Rows in train.csv format without Sales/Customers: Store, DayOfWeek, Date,
        Open, Promo, StateHoliday, SchoolHoliday
    store_metadata : pd.DataFrame, optional
        Store metadata (store.csv). If None, loaded from store_path
    store_path : str or Path
        Path to store metadata (used if store_metadata is None)

    Returns
    -------
    pd.DataFrame
        Every column (raw, store, and engineered), one row per input row, in input order
    """
    df = raw_data.copy()
    # The cleaning step expects Sales and Customers; no feature uses them
    for col in ["Sales", "Customers"]:
        if col not in df.columns:
            df[col] = 0
    # Cleaning sorts by Store and Date; remember the input order to restore it
    df["_input_order"] = range(len(df))

    if store_metadata is None:
        store_metadata = read_csv(store_path)

    clean_df = basic_cleaning(merge_store_info(df, store_metadata))
    featured_df = build_all_features(clean_df)
    featured_df = featured_df.sort_values("_input_order").drop(columns="_input_order")
    return featured_df.reset_index(drop=True)


def select_model_features(featured_df: pd.DataFrame) -> pd.DataFrame:
    """Select the model's feature columns (config/params.yaml) with training dtypes."""
    X = featured_df[get_model_features()].copy()
    for col in get_categorical_features():
        X[col] = X[col].astype(str).astype("category")
    return X


def prepare_prediction_data(
    raw_data: pd.DataFrame,
    store_metadata: pd.DataFrame | None = None,
    store_path: Union[str, Path] = STORE_PATH,
) -> pd.DataFrame:
    """Turn raw request rows into model-ready features.

    Parameters
    ----------
    raw_data : pd.DataFrame
        Rows with Store, DayOfWeek, Date, Open, Promo, StateHoliday, SchoolHoliday
    store_metadata : pd.DataFrame, optional
        Store metadata (store.csv). If None, loaded from store_path
    store_path : str or Path
        Path to store metadata (used if store_metadata is None)

    Returns
    -------
    pd.DataFrame
        The 12 model features, one row per input row, in input order

    Examples
    --------
    >>> raw_data = pd.DataFrame({
    ...     "Store": [1], "DayOfWeek": [5], "Date": ["2015-08-07"], "Open": [1],
    ...     "Promo": [1], "StateHoliday": ["0"], "SchoolHoliday": [0],
    ... })
    >>> prepare_prediction_data(raw_data).shape
    (1, 12)
    """
    logger.info(f"Preparing {len(raw_data)} records for prediction")
    return select_model_features(build_prediction_features(raw_data, store_metadata, store_path))


def prepare_prediction_data_from_csv(
    csv_path: str,
    store_metadata: pd.DataFrame | None = None,
    store_path: Union[str, Path] = STORE_PATH,
) -> pd.DataFrame:
    """Load raw data from CSV and prepare for predictions.

    Convenience wrapper around prepare_prediction_data() that loads data from CSV.

    Parameters
    ----------
    csv_path : str
        Path to CSV file with raw prediction data (train.csv format)
    store_metadata : pd.DataFrame, optional
        Store metadata. If None, will be loaded from store_path
    store_path : str or Path
        Path to store metadata file

    Returns
    -------
    pd.DataFrame
        Model-ready features
    """
    logger.info(f"Loading raw data from {csv_path}")
    raw_data = read_csv(csv_path)
    logger.info(f"Loaded {len(raw_data)} records")

    return prepare_prediction_data(raw_data, store_metadata, store_path)


def get_feature_columns() -> list[str]:
    """Return the model's feature column names (from config/params.yaml)."""
    return get_model_features()


def validate_input_data(
    raw_data: pd.DataFrame, known_stores: set[int] | None = None
) -> tuple[bool, list[str]]:
    """Validate that raw input data has required columns and types.

    Parameters
    ----------
    raw_data : pd.DataFrame
        Raw input data to validate
    known_stores : set of int, optional
        Store IDs the system has data for. If None, read from data/raw/store.csv

    Returns
    -------
    tuple of (bool, list of str)
        (is_valid, list of error messages)

    Examples
    --------
    >>> data = pd.DataFrame({"Store": [1], "DayOfWeek": [5]})
    >>> is_valid, errors = validate_input_data(data)
    >>> is_valid
    False
    >>> "Date" in errors[0]
    True
    """
    errors = []

    # Required columns
    required_cols = ["Store", "DayOfWeek", "Date", "Open", "Promo", "StateHoliday", "SchoolHoliday"]

    # Check for missing columns
    missing_cols = [col for col in required_cols if col not in raw_data.columns]
    if missing_cols:
        errors.append(f"Missing required columns: {missing_cols}")

    # Check data types and ranges (basic validation)
    if "Store" in raw_data.columns:
        if not pd.api.types.is_integer_dtype(raw_data["Store"]):
            errors.append("Store must be integer type")
        else:
            if known_stores is None:
                known_stores = get_known_stores()
            unknown = sorted(set(raw_data["Store"].astype(int)) - known_stores)
            if unknown:
                errors.append(
                    f"Unknown store ID(s) {unknown}: this system has data for "
                    f"{len(known_stores)} stores (see data/raw/store.csv)"
                )

    if "DayOfWeek" in raw_data.columns:
        if not pd.api.types.is_integer_dtype(raw_data["DayOfWeek"]):
            errors.append("DayOfWeek must be integer type")
        elif (raw_data["DayOfWeek"] < 1).any() or (raw_data["DayOfWeek"] > 7).any():
            errors.append("DayOfWeek must be between 1 and 7")

    if "Open" in raw_data.columns:
        if not raw_data["Open"].isin([0, 1]).all():
            errors.append("Open must be 0 or 1")

    if "Promo" in raw_data.columns:
        if not raw_data["Promo"].isin([0, 1]).all():
            errors.append("Promo must be 0 or 1")

    if "SchoolHoliday" in raw_data.columns:
        if not raw_data["SchoolHoliday"].isin([0, 1]).all():
            errors.append("SchoolHoliday must be 0 or 1")

    if "StateHoliday" in raw_data.columns:
        valid_holidays = ["0", "a", "b", "c"]
        if not raw_data["StateHoliday"].isin(valid_holidays).all():
            errors.append(f"StateHoliday must be one of {valid_holidays}")

    is_valid = len(errors) == 0
    return is_valid, errors


def main():
    """Example usage of prediction data preparation."""
    logger.info("=" * 70)
    logger.info("Prediction Data Preparation Example")
    logger.info("=" * 70)

    # Example 1: Prepare data from DataFrame
    sample_data = pd.DataFrame(
        {
            "Store": [1, 1, 21],
            "DayOfWeek": [5, 6, 5],
            "Date": ["2015-08-07", "2015-08-08", "2015-08-07"],
            "Open": [1, 1, 1],
            "Promo": [1, 0, 1],
            "StateHoliday": ["0", "0", "0"],
            "SchoolHoliday": [0, 0, 0],
        }
    )

    logger.info("\nValidating input data...")
    is_valid, errors = validate_input_data(sample_data)
    if is_valid:
        logger.info("✓ Input data is valid")
    else:
        logger.error(f"✗ Input data validation failed: {errors}")
        return

    logger.info("\nPreparing features...")
    features = prepare_prediction_data(sample_data)

    logger.info(f"\nPrepared features shape: {features.shape}")
    logger.info(f"Feature columns: {sorted(features.columns.tolist())}")

    logger.info("\n" + "=" * 70)
    logger.info("Example complete!")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
