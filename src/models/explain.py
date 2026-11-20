"""SHAP values for the Rossmann LightGBM model.

SHAP values split each prediction into one contribution per feature: the
prediction equals ``base_value`` (the average prediction) plus the sum of the
feature contributions. LightGBM computes them natively, so no extra package is
needed.

Example
-------
>>> from models.explain import compute_shap_values
>>> shap_df = compute_shap_values(model, X_holdout.head(5))
>>> shap_df.drop(columns="base_value").iloc[0].sort_values()
"""

import lightgbm as lgb
import pandas as pd


def compute_shap_values(model: lgb.Booster, X: pd.DataFrame) -> pd.DataFrame:
    """Compute SHAP values for each row of X.

    Parameters
    ----------
    model : lgb.Booster
        Trained LightGBM model
    X : pd.DataFrame
        Feature rows, with the same columns the model was trained on

    Returns
    -------
    pd.DataFrame
        One column per feature (its contribution to the prediction, in sales
        units) plus ``base_value``. Each row sums to the model's prediction.
    """
    contributions = model.predict(X, pred_contrib=True)
    columns = list(X.columns) + ["base_value"]
    return pd.DataFrame(contributions, columns=columns, index=X.index)
