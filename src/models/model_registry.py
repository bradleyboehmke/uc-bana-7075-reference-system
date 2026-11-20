"""MLflow Model Registry utilities for Rossmann forecasting.

This module provides helper functions for managing models in MLflow Model Registry, including
registration, promotion, and loading models.
"""

import logging
from typing import Any

import mlflow
import mlflow.pyfunc
from mlflow.lightgbm import log_model as log_lightgbm_model
from mlflow.tracking import MlflowClient

logger = logging.getLogger(__name__)


def get_mlflow_client() -> MlflowClient:
    """Get an MLflow tracking client.

    Returns
    -------
    MlflowClient
        Initialized MLflow client
    """
    return MlflowClient()


def register_model(
    model,
    model_name: str,
    description: str | None = None,
) -> str:
    """Log a trained LightGBM model to the active MLflow run and register it.

    Parameters
    ----------
    model : lightgbm.Booster
        Trained model
    model_name : str
        Name for the registered model (e.g., 'rossmann-model')
    description : str, optional
        Description for the new model version

    Returns
    -------
    str
        Model version number

    Examples
    --------
    >>> with mlflow.start_run():
    ...     version = register_model(model, model_name='rossmann-model')
    """
    logger.info(f"Registering model: {model_name}")
    model_info = log_lightgbm_model(
        model,
        name="model",
        registered_model_name=model_name,
    )
    version = str(model_info.registered_model_version)

    if description:
        client = get_mlflow_client()
        client.update_model_version(name=model_name, version=version, description=description)

    logger.info(f"✓ Registered {model_name} version {version}")
    return version


def promote_model(
    model_name: str,
    version: str,
    stage: str,
    archive_existing: bool = True,
) -> None:
    """Promote a model version to a specific stage.

    Parameters
    ----------
    model_name : str
        Name of the registered model
    version : str
        Model version to promote
    stage : str
        Target stage ('Staging', 'Production', or 'Archived')
    archive_existing : bool, default=True
        If True, archive existing models in the target stage

    Examples
    --------
    >>> # Promote to Staging
    >>> promote_model('rossmann-model', version='1', stage='Staging')
    >>> # Promote to Production
    >>> promote_model('rossmann-model', version='2', stage='Production')
    """
    client = get_mlflow_client()

    # Archive existing models in the target stage if requested
    if archive_existing and stage != "Archived":
        existing_versions = client.get_latest_versions(model_name, stages=[stage])
        for mv in existing_versions:
            logger.info(f"Archiving {model_name} version {mv.version} (was in {stage})")
            client.transition_model_version_stage(
                name=model_name, version=mv.version, stage="Archived"
            )

    # Promote the specified version
    logger.info(f"Promoting {model_name} version {version} to {stage}")
    client.transition_model_version_stage(name=model_name, version=version, stage=stage)
    logger.info(f"✓ {model_name} version {version} is now in {stage}")


def get_model_version(model_name: str, stage: str = "Production") -> str | None:
    """Get the version number of a model in a specific stage.

    Parameters
    ----------
    model_name : str
        Name of the registered model
    stage : str, default='Production'
        Model stage ('Staging', 'Production', or 'None')

    Returns
    -------
    str or None
        Model version number, or None if no model in stage

    Examples
    --------
    >>> version = get_model_version('rossmann-model', stage='Production')
    >>> print(f"Production version: {version}")
    """
    client = get_mlflow_client()
    versions = client.get_latest_versions(model_name, stages=[stage])

    if not versions:
        logger.warning(f"No model found for {model_name} in {stage} stage")
        return None

    version = versions[0].version
    logger.info(f"{model_name} {stage} version: {version}")
    return version


def load_model(model_name: str, stage: str = "Production"):
    """Load a model from MLflow Model Registry.

    Parameters
    ----------
    model_name : str
        Name of the registered model
    stage : str, default='Production'
        Model stage to load ('Staging', 'Production', or version number)

    Returns
    -------
    mlflow.pyfunc.PyFuncModel
        Loaded model ready for predictions

    Examples
    --------
    >>> # Load production model
    >>> model = load_model('rossmann-model', stage='Production')
    >>> predictions = model.predict(X_test)
    >>>
    >>> # Load specific version
    >>> model = load_model('rossmann-model', stage='1')
    >>> predictions = model.predict(X_test)
    """
    if stage in ["Staging", "Production", "Archived"]:
        model_uri = f"models:/{model_name}/{stage}"
    else:
        # Assume it's a version number
        model_uri = f"models:/{model_name}/{stage}"

    logger.info(f"Loading model from: {model_uri}")
    model = mlflow.pyfunc.load_model(model_uri)
    logger.info("✓ Model loaded successfully")

    return model


def get_model_info(model_name: str, version: str | None = None) -> dict[str, Any]:
    """Get detailed information about a registered model.

    Parameters
    ----------
    model_name : str
        Name of the registered model
    version : str, optional
        Model version. If None, gets info for all versions

    Returns
    -------
    dict
        Model information including version, stage, metrics, etc.

    Examples
    --------
    >>> info = get_model_info('rossmann-model', version='1')
    >>> print(f"Model stage: {info['current_stage']}")
    >>> print(f"Run ID: {info['run_id']}")
    """
    client = get_mlflow_client()

    if version:
        mv = client.get_model_version(name=model_name, version=version)
        return {
            "name": mv.name,
            "version": mv.version,
            "current_stage": mv.current_stage,
            "description": mv.description,
            "run_id": mv.run_id,
            "status": mv.status,
            "creation_timestamp": mv.creation_timestamp,
            "last_updated_timestamp": mv.last_updated_timestamp,
        }
    else:
        # Get all versions
        model = client.get_registered_model(model_name)
        versions = client.search_model_versions(f"name='{model_name}'")

        return {
            "name": model.name,
            "description": model.description,
            "creation_timestamp": model.creation_timestamp,
            "last_updated_timestamp": model.last_updated_timestamp,
            "versions": [
                {
                    "version": mv.version,
                    "stage": mv.current_stage,
                    "run_id": mv.run_id,
                    "status": mv.status,
                }
                for mv in versions
            ],
        }


def list_registered_models() -> list:
    """List all registered models in MLflow Model Registry.

    Returns
    -------
    list
        List of registered model names

    Examples
    --------
    >>> models = list_registered_models()
    >>> for model_name in models:
    ...     print(model_name)
    """
    client = get_mlflow_client()
    registered_models = client.search_registered_models()

    model_names = [rm.name for rm in registered_models]
    logger.info(f"Found {len(model_names)} registered models")

    return model_names
