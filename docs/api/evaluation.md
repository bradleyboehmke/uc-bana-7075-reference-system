# Evaluation API

`src/evaluation/metrics.py` provides the metrics used to score the model.

## How the model is evaluated

Training holds out the last 6 weeks of data (`training.holdout_days` in `config/params.yaml`), trains on everything before it, and scores predictions on the holdout. This simulates forecasting the next six weeks. Only open-store days are scored.

`validate_model.py` repeats this scoring for a registered model version and promotes it to Staging if RMSPE is at or below `validation.max_rmspe`.

## Module Reference

::: src.evaluation.metrics
    options:
      show_source: true
      heading_level: 3
      show_root_heading: false

## Metrics

### RMSPE (Root Mean Square Percentage Error)

The primary metric for Rossmann forecasting:

$$ \\text{RMSPE} = \\sqrt{\\frac{1}{n}\\sum\_{i=1}^{n}\\left(\\frac{y_i - \\hat{y}\_i}{y_i}\\right)^2} $$

where $y_i$ is actual sales, $\\hat{y}\_i$ is predicted sales, and $n$ counts observations with $y_i \\neq 0$.

**Why RMSPE?**

- It measures percentage error, so a store selling 3,000 a day and one selling 15,000 count equally
- It ignores days with zero sales

An RMSPE of 0.16 means predictions are typically about 16% off.

```python
from evaluation.metrics import mae, rmse, rmspe

rmspe(y_true, y_pred)  # primary metric
rmse(y_true, y_pred)   # penalizes large errors, in sales units
mae(y_true, y_pred)    # average absolute error, in sales units
```

## Explaining predictions

Feature contributions for individual predictions (SHAP values) are in `src/models/explain.py`; see the [Models API](models.md).

## Related Modules

- [Models](models.md) - Training and validation that use these metrics
- [Monitoring](monitoring.md) - Production model evaluation
