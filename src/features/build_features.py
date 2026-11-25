"""Feature engineering for the Rossmann forecasting project.

The model uses 12 features in four groups. The list lives in config/params.yaml
(``features.model_features``); this module creates the ones that are not already
columns in the cleaned data.

- When: DayOfWeek (raw), Month, Week, Year (created here)
- Store: StoreType, Assortment, CompetitionDistance (raw, from store.csv)
- Promotions: Promo, Promo2 (raw), Promo2Active (created here)
- Holidays: StateHoliday, SchoolHoliday (raw)

Every feature is known in advance for any future date, so the API can build the
same features from a request as training builds from history. There are no
features computed from past sales.
"""

import sys
from pathlib import Path

import pandas as pd

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from utils.io import load_params, read_parquet, save_parquet
from utils.log import get_logger

logger = get_logger(__name__)


def get_model_features() -> list[str]:
    """Return the model's feature columns, as listed in config/params.yaml."""
    return list(load_params()["features"]["model_features"])


def get_categorical_features() -> list[str]:
    """Return the categorical feature columns, as listed in config/params.yaml."""
    return list(load_params()["features"]["categorical_features"])


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add Year, Month, and Week (ISO week of year) derived from Date.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with a datetime Date column

    Returns
    -------
    pd.DataFrame
        Dataframe with Year, Month, and Week added
    """
    df = df.copy()
    df["Year"] = df["Date"].dt.year.astype("int16")
    df["Month"] = df["Date"].dt.month.astype("int16")
    df["Week"] = df["Date"].dt.isocalendar().week.astype("int16")
    return df


def add_promo_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add Promo2Active: whether the store's ongoing Promo2 coupon program runs this month.

    A store in Promo2 restarts its coupons in the months listed in PromoInterval
    (for example "Jan,Apr,Jul,Oct").

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with Date, Promo2, and PromoInterval columns

    Returns
    -------
    pd.DataFrame
        Dataframe with Promo2Active added (1 = active this month, 0 = not)
    """
    df = df.copy()
    months = df["Date"].dt.strftime("%b")
    # store.csv spells September "Sept"; normalize to "Sep" to match strftime
    intervals = df["PromoInterval"].astype(str).str.replace("Sept", "Sep")
    in_interval = [month in interval for month, interval in zip(months, intervals, strict=True)]
    df["Promo2Active"] = ((df["Promo2"] == 1) & pd.Series(in_interval, index=df.index)).astype(
        "int8"
    )
    return df


def build_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create every derived feature the model needs.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataframe (output of src.data.make_dataset)

    Returns
    -------
    pd.DataFrame
        The input columns plus Year, Month, Week, and Promo2Active
    """
    logger.info("Building features")
    df = add_calendar_features(df)
    df = add_promo_features(df)
    logger.info(f"Model features: {get_model_features()}")
    return df


def main():
    """Build features from data/processed/train_clean.parquet."""
    df = read_parquet("data/processed/train_clean.parquet")
    logger.info(f"Loaded {len(df):,} rows, {len(df.columns)} columns")

    df_featured = build_all_features(df)

    output_path = "data/processed/train_features.parquet"
    save_parquet(df_featured, output_path)
    logger.info(f"Saved {len(df_featured):,} rows to {output_path}")


if __name__ == "__main__":
    main()
