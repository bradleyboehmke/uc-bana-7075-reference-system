"""Train the Rossmann sales model: one LightGBM model with fixed hyperparameters.

Steps:

1. Load data/processed/train_features.parquet.
2. Hold out the last 6 weeks (``training.holdout_days``) for evaluation.
3. Train LightGBM on the earlier data, using the hyperparameters in
   config/params.yaml (``model.lightgbm``). There is no tuning.
4. Measure RMSPE, RMSE, and MAE on the holdout.
5. Log parameters, metrics, and the data version to MLflow and register the model
   under ``mlflow.model_name``.

Run from the project root:

    python -m src.models.train
"""

import sys
import time
from pathlib import Path

import lightgbm as lgb
import mlflow
import pandas as pd

# Add src/ to path so the module runs as `python -m src.models.train` or as a script
sys.path.append(str(Path(__file__).parent.parent))

from evaluation.metrics import mae, rmse, rmspe
from features.build_features import get_categorical_features, get_model_features
from models.model_registry import register_model
from utils.io import load_params, read_parquet
from utils.log import get_logger
from utils.mlflow_utils import log_dvc_data_version, setup_mlflow

logger = get_logger(__name__)

DATA_PATH = "data/processed/train_features.parquet"


def split_holdout(df: pd.DataFrame, holdout_days: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split data into training data and a holdout of the last ``holdout_days`` days.

    Parameters
    ----------
    df : pd.DataFrame
        Data with a Date column
    holdout_days : int, default=42
        Length of the holdout period in days

    Returns
    -------
    tuple of pd.DataFrame
        (train_df, holdout_df)
    """
    holdout_start = df["Date"].max() - pd.Timedelta(days=holdout_days - 1)
    train_df = df[df["Date"] < holdout_start].copy()
    holdout_df = df[df["Date"] >= holdout_start].copy()
    logger.info(
        f"Train: {len(train_df):,} rows to {train_df['Date'].max().date()}; "
        f"holdout: {len(holdout_df):,} rows from {holdout_start.date()}"
    )
    return train_df, holdout_df


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Select open-store rows and the model features.

    Closed stores sell nothing, so the model only learns from (and is scored on)
    days when a store was open.

    Parameters
    ----------
    df : pd.DataFrame
        Featured data including Open and Sales

    Returns
    -------
    tuple
        (X, y): the feature matrix and Sales
    """
    open_days = df[df["Open"] == 1]
    X = open_days[get_model_features()].copy()
    for col in get_categorical_features():
        X[col] = X[col].astype(str).astype("category")
    y = open_days["Sales"]
    return X, y


def train_model(X: pd.DataFrame, y: pd.Series, params: dict | None = None) -> lgb.Booster:
    """Train LightGBM with fixed hyperparameters.

    Parameters
    ----------
    X : pd.DataFrame
        Training features
    y : pd.Series
        Training target (Sales)
    params : dict, optional
        LightGBM parameters including ``num_boost_round``.
        Defaults to ``model.lightgbm`` in config/params.yaml.

    Returns
    -------
    lgb.Booster
        Trained model
    """
    params = dict(params or load_params()["model"]["lightgbm"])
    num_boost_round = params.pop("num_boost_round")
    dataset = lgb.Dataset(X, label=y, categorical_feature=get_categorical_features())
    return lgb.train(params, dataset, num_boost_round=num_boost_round)


def evaluate(model: lgb.Booster, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    """Score the model on holdout data.

    Returns
    -------
    dict
        rmspe, rmse, and mae
    """
    predictions = model.predict(X)
    return {
        "rmspe": float(rmspe(y.values, predictions)),
        "rmse": float(rmse(y.values, predictions)),
        "mae": float(mae(y.values, predictions)),
    }


def main(data_path: str = DATA_PATH, register: bool = True) -> dict:
    """Train, evaluate, log to MLflow, and register the model.

    Parameters
    ----------
    data_path : str
        Path to the featured data
    register : bool, default=True
        Register the trained model in the MLflow Model Registry

    Returns
    -------
    dict
        metrics, train_seconds, run_id, and version (None if not registered)
    """
    config = load_params()
    lgb_params = config["model"]["lightgbm"]
    model_name = config["mlflow"]["model_name"]

    setup_mlflow()
    with mlflow.start_run(run_name="train_lightgbm") as run:
        log_dvc_data_version(data_path)

        df = read_parquet(data_path)
        train_df, holdout_df = split_holdout(df, config["training"]["holdout_days"])
        X_train, y_train = prepare_features(train_df)
        X_holdout, y_holdout = prepare_features(holdout_df)

        mlflow.log_params(lgb_params)
        mlflow.log_param("features", ",".join(get_model_features()))
        mlflow.log_param("n_features", X_train.shape[1])
        mlflow.log_param("train_rows", len(X_train))
        mlflow.log_param("holdout_rows", len(X_holdout))

        start = time.time()
        model = train_model(X_train, y_train, lgb_params)
        train_seconds = time.time() - start
        logger.info(f"Trained in {train_seconds:.1f}s")

        metrics = evaluate(model, X_holdout, y_holdout)
        mlflow.log_metrics({**metrics, "train_seconds": train_seconds})
        logger.info(
            f"Holdout RMSPE {metrics['rmspe']:.4f}, RMSE {metrics['rmse']:.0f}, "
            f"MAE {metrics['mae']:.0f}"
        )

        version = None
        if register:
            version = register_model(
                model,
                model_name=model_name,
                description=f"LightGBM, holdout RMSPE {metrics['rmspe']:.4f}",
            )

    return {
        "metrics": metrics,
        "train_seconds": train_seconds,
        "run_id": run.info.run_id,
        "version": version,
    }


if __name__ == "__main__":
    result = main()
    print(f"\nHoldout RMSPE: {result['metrics']['rmspe']:.4f}")
    print(f"Registered model version: {result['version']}")
    print("\nNext: python src/models/validate_model.py")
