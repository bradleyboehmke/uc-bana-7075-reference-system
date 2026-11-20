# Feature Engineering API

`src/features/build_features.py` creates the features the model needs from the cleaned data.

## Overview

The model uses 12 features in four groups. Most are columns already in the raw data; this module creates four of them (Year, Month, Week, Promo2Active).

| Group | Features | Source |
|---|---|---|
| When | `DayOfWeek` | train.csv |
| | `Month`, `Week`, `Year` | Created from `Date` |
| Store | `StoreType`, `Assortment`, `CompetitionDistance` | store.csv |
| Promotions | `Promo` | train.csv |
| | `Promo2` | store.csv |
| | `Promo2Active` | Created from `Date`, `Promo2`, `PromoInterval` |
| Holidays | `StateHoliday`, `SchoolHoliday` | train.csv |

The authoritative list is `features.model_features` in `config/params.yaml`; `get_model_features()` returns it.

Every feature is known in advance for any future date. None is computed from past sales, so the features the API builds for a request match the ones training used.

## Module Reference

::: src.features.build_features
    options:
      show_source: true
      heading_level: 3
      show_root_heading: false

## Usage Examples

```python
from features.build_features import build_all_features, get_model_features
from utils.io import read_parquet

df = read_parquet("data/processed/train_clean.parquet")
df = build_all_features(df)
X = df[get_model_features()]
```

From the command line:

```bash
python -m src.features.build_features
# Reads data/processed/train_clean.parquet
# Writes data/processed/train_features.parquet
```

## Created features

### Calendar: `add_calendar_features()`

- `Year`: calendar year
- `Month`: 1–12
- `Week`: ISO week of the year, 1–53

### Promotions: `add_promo_features()`

- `Promo2Active`: 1 when the store takes part in Promo2 **and** the current month is listed in its `PromoInterval` (for example `"Jan,Apr,Jul,Oct"`), else 0

## Data Flow

```mermaid
graph LR
    A[train_clean.parquet] --> B[add_calendar_features]
    B --> C[add_promo_features]
    C --> D[train_features.parquet]
```

## Related Modules

- [Data API](data.md) - Loading and cleaning the raw data
- [Models API](models.md) - Training on these features
