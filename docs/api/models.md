# Models API

The `src/models/` package trains, registers, validates, serves, and explains one LightGBM model with fixed hyperparameters. See [Model Training](../modelops/training.md) for the features and the workflow.

| Module | Purpose |
|---|---|
| `train.py` | Train, evaluate on the 6-week holdout, log to MLflow, register |
| `model_registry.py` | Register, promote, load, and describe model versions |
| `validate_model.py` | Score a version on the holdout; promote to Staging if it passes |
| `predict.py` | Batch predictions with a model loaded by stage or version |
| `explain.py` | SHAP values for individual predictions |

## Module Reference

### Training Pipeline

::: src.models.train
    options:
      show_source: true
      heading_level: 4
      show_root_heading: false

### Model Registry

::: src.models.model_registry
    options:
      show_source: true
      heading_level: 4
      show_root_heading: false

### Model Validation

::: src.models.validate_model
    options:
      show_source: true
      heading_level: 4
      show_root_heading: false

### Inference Pipeline

::: src.models.predict
    options:
      show_source: true
      heading_level: 4
      show_root_heading: false

### Explanations (SHAP)

::: src.models.explain
    options:
      show_source: true
      heading_level: 4
      show_root_heading: false

## Usage Examples

### Train from the command line

```bash
python -m src.models.train
```

### Train from Python

```python
from models.train import evaluate, prepare_features, split_holdout, train_model
from utils.io import read_parquet

df = read_parquet("data/processed/train_features.parquet")
train_df, holdout_df = split_holdout(df, holdout_days=42)

X_train, y_train = prepare_features(train_df)
X_holdout, y_holdout = prepare_features(holdout_df)

model = train_model(X_train, y_train)  # hyperparameters from config/params.yaml
print(evaluate(model, X_holdout, y_holdout))  # {'rmspe': ..., 'rmse': ..., 'mae': ...}
```

### Explain a prediction

```python
from models.explain import compute_shap_values

shap_df = compute_shap_values(model, X_holdout.head(1))
shap_df.drop(columns="base_value").iloc[0].sort_values()
```

### Load the Production model

```python
from models.model_registry import load_model

model = load_model("rossmann-model", stage="Production")
predictions = model.predict(X_holdout)
```

## Related Modules

- [Features API](features.md) - Builds the model's 12 features
- [Evaluation API](evaluation.md) - RMSPE and other metrics
