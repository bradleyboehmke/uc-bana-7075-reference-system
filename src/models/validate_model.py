"""Model validation and promotion for Rossmann forecasting.

This module validates a candidate model against performance thresholds and promotes it to Staging if
it passes all checks.
"""

import logging

import numpy as np
import pandas as pd

from evaluation.metrics import rmspe
from features.build_features import get_model_features
from models.model_registry import (
    get_model_version,
    load_model,
    promote_model,
)
from models.train import prepare_features, split_holdout
from utils.io import load_params, read_parquet

logger = logging.getLogger(__name__)


def load_holdout_data(
    data_path: str = "data/processed/train_features.parquet",
    holdout_days: int = 42,
) -> tuple:
    """Load holdout validation data.

    Parameters
    ----------
    data_path : str
        Path to processed features
    holdout_days : int
        Number of days for holdout set

    Returns
    -------
    tuple
        (X_holdout, y_holdout, feature_cols)
    """
    logger.info(f"Loading holdout data from: {data_path}")
    df = read_parquet(data_path)
    _, holdout_df = split_holdout(df, holdout_days)
    X_holdout, y = prepare_features(holdout_df)
    y_holdout = y.values
    feature_cols = get_model_features()

    logger.info(f"Holdout data: {len(X_holdout):,} rows (open stores)")

    return X_holdout, y_holdout, feature_cols


def evaluate_model(model, X_test: pd.DataFrame, y_test: np.ndarray) -> dict[str, float]:
    """Evaluate model on test data.

    Parameters
    ----------
    model : mlflow.pyfunc.PyFuncModel
        Loaded model
    X_test : pd.DataFrame
        Test features
    y_test : np.ndarray
        Test targets

    Returns
    -------
    dict
        Evaluation metrics
    """
    logger.info("Generating predictions...")
    predictions = model.predict(X_test)

    logger.info("Calculating metrics...")
    metrics = {
        "rmspe": float(rmspe(y_test, predictions)),
        "rmse": float(np.sqrt(np.mean((y_test - predictions) ** 2))),
        "mae": float(np.mean(np.abs(y_test - predictions))),
        "mape": float(np.mean(np.abs((y_test - predictions) / y_test)) * 100),
    }

    logger.info("Model Performance:")
    logger.info(f"  RMSPE: {metrics['rmspe']:.6f}")
    logger.info(f"  RMSE:  {metrics['rmse']:.2f}")
    logger.info(f"  MAE:   {metrics['mae']:.2f}")
    logger.info(f"  MAPE:  {metrics['mape']:.2f}%")

    return metrics


def check_performance_threshold(
    metrics: dict[str, float],
    threshold_rmspe: float | None = None,
) -> bool:
    """Check if model meets performance thresholds.

    Parameters
    ----------
    metrics : dict
        Model evaluation metrics
    threshold_rmspe : float, optional
        Maximum acceptable RMSPE. Defaults to ``validation.max_rmspe`` in config/params.yaml.

    Returns
    -------
    bool
        True if model passes threshold checks
    """
    if threshold_rmspe is None:
        threshold_rmspe = load_params()["validation"]["max_rmspe"]

    logger.info("=" * 70)
    logger.info("PERFORMANCE THRESHOLD CHECKS")
    logger.info("=" * 70)

    # Check RMSPE
    rmspe_pass = metrics["rmspe"] <= threshold_rmspe
    logger.info(
        f"RMSPE: {metrics['rmspe']:.6f} <= {threshold_rmspe:.6f} "
        f"{'✓ PASS' if rmspe_pass else '✗ FAIL'}"
    )

    # Check for reasonable predictions (not extreme values)
    predictions_reasonable = metrics["mae"] < 2000  # Average error less than $2000 seems reasonable
    logger.info(
        f"MAE: {metrics['mae']:.2f} < 2000 " f"{'✓ PASS' if predictions_reasonable else '✗ FAIL'}"
    )

    all_pass = rmspe_pass and predictions_reasonable

    logger.info("=" * 70)
    if all_pass:
        logger.info("✓ ALL CHECKS PASSED")
    else:
        logger.info("✗ VALIDATION FAILED")
    logger.info("=" * 70)

    return all_pass


def validate_and_promote(
    model_name: str | None = None,
    version: str | None = None,
    data_path: str = "data/processed/train_features.parquet",
    threshold_rmspe: float | None = None,
    auto_promote: bool = True,
) -> dict[str, any]:
    """Validate a model and promote to Staging if passing.

    Parameters
    ----------
    model_name : str
        Name of registered model
    version : str, optional
        Model version to validate. If None, uses latest version
    data_path : str
        Path to processed data
    threshold_rmspe : float, optional
        Maximum acceptable RMSPE. Defaults to ``validation.max_rmspe`` in config/params.yaml.
    auto_promote : bool
        Automatically promote to Staging if passing

    Returns
    -------
    dict
        Validation results including metrics and promotion status
    """
    model_name = model_name or load_params()["mlflow"]["model_name"]
    logger.info("=" * 70)
    logger.info("MODEL VALIDATION & PROMOTION")
    logger.info("=" * 70)

    # Get version if not specified
    if version is None:
        # Find the latest version (any stage)
        from mlflow.tracking import MlflowClient

        client = MlflowClient()
        versions = client.search_model_versions(f"name='{model_name}'")
        if not versions:
            raise ValueError(f"No versions found for model: {model_name}")
        version = max([int(mv.version) for mv in versions])
        logger.info(f"Using latest version: {version}")
    else:
        logger.info(f"Validating version: {version}")

    # Load model
    logger.info(f"\nLoading model: {model_name} version {version}")
    model = load_model(model_name, stage=str(version))

    # Load holdout data
    logger.info("\nLoading holdout validation data...")
    X_holdout, y_holdout, feature_cols = load_holdout_data(data_path)

    # Evaluate model
    logger.info("\nEvaluating model performance...")
    metrics = evaluate_model(model, X_holdout, y_holdout)

    # Check thresholds
    logger.info("\nChecking performance thresholds...")
    if threshold_rmspe is None:
        threshold_rmspe = load_params()["validation"]["max_rmspe"]
    passed = check_performance_threshold(metrics, threshold_rmspe)

    # Promotion decision
    promoted = False
    if passed and auto_promote:
        logger.info("\n✓ Model passed validation, promoting to Staging...")
        promote_model(model_name, version=str(version), stage="Staging")
        promoted = True
        logger.info(f"✓ Model {model_name} v{version} promoted to Staging")
    elif passed:
        logger.info("\n✓ Model passed validation (auto_promote=False, manual promotion required)")
    else:
        logger.info("\n✗ Model did not pass validation, not promoting")

    results = {
        "model_name": model_name,
        "version": version,
        "metrics": metrics,
        "threshold_rmspe": threshold_rmspe,
        "passed": passed,
        "promoted": promoted,
        "stage": "Staging" if promoted else "None",
    }

    logger.info("=" * 70)
    logger.info("VALIDATION SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Model: {model_name} v{version}")
    logger.info(f"RMSPE: {metrics['rmspe']:.6f}")
    logger.info(f"Passed: {passed}")
    logger.info(f"Promoted to Staging: {promoted}")
    logger.info("=" * 70)

    return results


def promote_to_production(
    model_name: str | None = None,
    version: str | None = None,
) -> None:
    """Promote a Staging model to Production.

    This should typically be done manually after testing the Staging model.

    Parameters
    ----------
    model_name : str
        Name of registered model
    version : str, optional
        Model version to promote. If None, uses current Staging version
    """
    model_name = model_name or load_params()["mlflow"]["model_name"]
    logger.info("=" * 70)
    logger.info("PROMOTE TO PRODUCTION")
    logger.info("=" * 70)

    # Get Staging version if not specified
    if version is None:
        version = get_model_version(model_name, stage="Staging")
        if version is None:
            raise ValueError(f"No Staging version found for {model_name}")
        logger.info(f"Promoting Staging version: {version}")

    # Promote to Production
    logger.info(f"\nPromoting {model_name} v{version} to Production...")
    promote_model(model_name, version=version, stage="Production")

    logger.info("=" * 70)
    logger.info(f"✓ {model_name} v{version} is now in Production")
    logger.info("=" * 70)


def main():
    """Main validation workflow."""
    import argparse

    parser = argparse.ArgumentParser(description="Validate and promote the Rossmann model")
    parser.add_argument(
        "--model-name", default=None, help="Model name in registry (default: from config)"
    )
    parser.add_argument("--version", type=str, help="Model version to validate")
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="RMSPE threshold (default: validation.max_rmspe in config/params.yaml)",
    )
    parser.add_argument(
        "--no-auto-promote",
        action="store_true",
        help="Disable automatic promotion to Staging",
    )
    parser.add_argument(
        "--promote-to-production",
        action="store_true",
        help="Promote Staging model to Production (manual step)",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    if args.promote_to_production:
        promote_to_production(model_name=args.model_name, version=args.version)
    else:
        results = validate_and_promote(
            model_name=args.model_name,
            version=args.version,
            threshold_rmspe=args.threshold,
            auto_promote=not args.no_auto_promote,
        )

        # Print summary
        print("\n" + "=" * 70)
        print("VALIDATION RESULTS")
        print("=" * 70)
        print(f"Model: {results['model_name']} v{results['version']}")
        print(f"RMSPE: {results['metrics']['rmspe']:.6f}")
        print(f"Threshold: {results['threshold_rmspe']:.6f}")
        print(f"Passed: {'✓ YES' if results['passed'] else '✗ NO'}")
        print(f"Stage: {results['stage']}")
        print("=" * 70)

        if results["passed"] and results["promoted"]:
            print("\n✓ Model validated and promoted to Staging!")
            print("\nNext steps:")
            print("  1. Test Staging model in production-like environment")
            print("  2. If satisfied, promote to Production:")
            print(
                f"     python src/models/validate_model.py --promote-to-production --version {results['version']}"
            )
        elif results["passed"]:
            print("\n✓ Model passed validation!")
            print("\nTo promote to Staging:")
            print(f"  python src/models/validate_model.py --version {results['version']}")
        else:
            print("\n✗ Model did not pass validation")
            print("\nConsider:")
            print("  - Retraining with more data")
            print("  - Revisiting the features or hyperparameters in config/params.yaml")


if __name__ == "__main__":
    main()
