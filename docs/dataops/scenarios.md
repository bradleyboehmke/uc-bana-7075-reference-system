# Real-World Scenarios

This page walks through practical scenarios showing how the DataOps pipeline handles different situations in production.

______________________________________________________________________

## Scenario 1: Initial Setup

**Context:** Setting up the DataOps pipeline for the first time with historical data.

### Week 1: Baseline Establishment

```bash
# Run DataOps pipeline on historical data (2013-2015)
bash scripts/dataops_workflow.sh

# Expected output:
# ✓ Raw data validation: PASSED
# ✓ Processed 1,017,209 records
# ✓ Created 32 standard features
# ✓ All validations: PASSED
```

**Results:**

- Clean processed data: `data/processed/train_clean.parquet`
- Feature dataset: `data/processed/train_features.parquet`
- DVC metadata: `*.dvc` files committed to Git
- Ready for model training

### Train Initial Model

```bash
# Train and register the model
python -m src.models.train

# Model achieves RMSPE = 0.16 on the holdout
# Deploy to production
```

______________________________________________________________________

## Scenario 2: New Data Arrives (Success Path)

**Context:** New week of sales data arrives from Rossmann's internal systems.

### Week 2: Routine Data Update

```bash
# New data file arrives
cp rossmann_sales_2015_week32.csv data/raw/train.csv

# Run DataOps workflow (automated via cron or Airflow)
bash scripts/dataops_workflow.sh
```

### Pipeline Automatically

1. **Validates new data** - Catches any schema changes
1. **Merges with historical data** - Combines with existing records
1. **Re-engineers all features** - Builds features for the new rows
1. **Versions new dataset** - Creates new DVC snapshot
1. **Triggers model retraining** - Signals ModelOps workflow

### Model Retraining

```bash
# Automatic trigger (via Airflow/GitHub Actions)
python -m src.models.train

# New model trained on expanded dataset
# Achieves RMSPE = 0.15 (improvement!)
# Automatically deployed to production
```

**Key Benefits:**

- ✅ Fully automated end-to-end
- ✅ Quality gates ensure data integrity
- ✅ Model continuously improves with more data
- ✅ Zero manual intervention required

______________________________________________________________________

## Scenario 3: Bad Data Arrives (Failure Path)

**Context:** New data file has quality issues that violate expectations.

### Week 3: Schema Change Detected

```bash
# New data file has schema issue (missing 'Store' column)
cp bad_data_missing_column.csv data/raw/train.csv

# Run DataOps workflow
bash scripts/dataops_workflow.sh
```

### Pipeline Response

```
============================================================
Step 1: Validating raw data...
============================================================
raw_store  PASSED  (8 of 8 rules passed)
raw_train  FAILED  (10 of 14 rules passed)
  ✗ expect_table_columns_to_match_ordered_list: found ['DayOfWeek', 'Date', 'Sales', 'Customers', ...]
      Rule: The file has the columns the pipeline reads, in the expected order.
  ...

✗ Data checks failed. Fix the data before running the next pipeline step.
Exit code: 1
```

**What Happens:**

1. ❌ Validation fails at Step 1
1. 🛑 Pipeline stops immediately
1. 📧 Alert sent to data engineering team
1. ✅ Production model continues running on last known good data
1. 👥 Team investigates and fixes data source

**Critical Point:** Bad data **never enters** the pipeline. This fail-safe behavior prevents broken models from reaching production.

______________________________________________________________________

## Scenario 4: Data Quality Degradation

**Context:** Data passes schema validation but has statistical anomalies.

### Week 4: Outlier Detection

```bash
# New data has unusual sales patterns
cp data_with_outliers.csv data/raw/train.csv

# Run DataOps workflow
bash scripts/dataops_workflow.sh
```

### Validation Catches Outliers

```
============================================================
Step 1: Validating raw data...
============================================================
raw_store  PASSED  (8 of 8 rules passed)
raw_train  FAILED  (13 of 14 rules passed)
  ✗ expect_column_values_to_be_between on Sales: 3 rows failed, for example [125000, 110000, 150000]
      Rule: Daily sales are never negative and no store sells over 50,000 a day.

✗ Data checks failed. Fix the data before running the next pipeline step.
```

**Response Options:**

1. **Investigate outliers**

    ```bash
    # Review outlier records
    python -c "
    import pandas as pd
    df = pd.read_csv('data/raw/train.csv')
    print(df[df['Sales'] > 50000])
    "
    ```

1. **Update the rule** (if legitimate)

    ```python
    # In src/data/expectations.py, if the spike is real (say why in the note)
    gxe.ExpectColumnValuesToBeBetween(
        column="Sales",
        min_value=0,
        max_value=200_000,
        notes="Raised from 50,000: holiday sales can exceed it.",
    )
    ```

1. **Cap outliers** (if data errors)

    ```python
    # In make_dataset.py
    df["Sales"] = df["Sales"].clip(upper=50000)
    ```

______________________________________________________________________

## Scenario 5: Feature Engineering Update

**Context:** Data scientist wants to add new experimental features.

### Developer Iteration

```bash
# Modify feature engineering code
vim src/features/build_features.py

# Add new experimental features (e.g., holiday interactions)
# Test locally
python -m src.features.build_features

# Run DVC pipeline (only reruns changed stages)
dvc repro
```

### DVC Smart Caching

```
Running stage 'build_features':
⚙️  Re-running build_features (code changed)
⚙️  Re-running validate_features (input changed)
✅ Skipping validate_raw (unchanged)
✅ Skipping process_data (unchanged)
✅ Skipping validate_processed (unchanged)

Completed in 45 seconds (vs. 5 minutes full rebuild)
```

**Benefits:**

- ⚡ Fast iteration (only reruns what changed)
- ✅ Validation ensures new features are valid
- 🔄 Easy rollback if features don't improve model

______________________________________________________________________

## Scenario 6: Rollback to Previous Data Version

**Context:** New data processing introduced a bug. Need to revert.

### Problem Detection

```bash
# Week 5: Model performance suddenly drops
# RMSPE increased from 0.097 → 0.125
# Suspect data processing bug in latest run
```

### Rollback with DVC

```bash
# List DVC commits
git log --oneline data/processed/train_features.parquet.dvc

# Output:
# a1b2c3d data: version features with new holiday interactions
# e4f5g6h data: version processed data and features (← this one worked!)
# i7j8k9l data: initial feature engineering

# Checkout previous version
git checkout e4f5g6h data/processed/train_features.parquet.dvc

# Pull previous data from DVC cache
dvc checkout

# Retrain model on previous data
python -m src.models.train

# Model performance restored: RMSPE = 0.16 ✅
```

**Next Steps:**

- Debug feature engineering code
- Fix bug
- Re-run pipeline
- Validate performance before deploying

______________________________________________________________________

## Scenario 7: Scheduled Pipeline Execution

**Context:** Production setup with weekly automated data refreshes.

### Cron Schedule

```bash
# crontab -e
# Run DataOps pipeline every Monday at 2 AM
0 2 * * 1 cd /path/to/rossmann-forecasting && bash scripts/dataops_workflow.sh >> logs/dataops.log 2>&1
```

### Typical Week

**Monday 2:00 AM:**

- Cron triggers DataOps pipeline
- New sales data processed
- Features engineered
- Data versioned with DVC

**Monday 3:00 AM:**

- Model retraining triggered
- New model evaluated
- If better: deployed to production
- If worse: old model retained

**Monday 4:00 AM:**

- Monitoring dashboards updated
- Team receives summary email
- System ready for the week

______________________________________________________________________

## Scenario 8: Multi-Environment Setup

**Context:** Separate pipelines for development, staging, and production.

### Environment Structure

```bash
# Development: Experiment with new features
dev/
├── data/raw/train_sample.csv  # 10% sample for fast iteration
└── .dvc/config  # Points to dev S3 bucket

# Staging: Test full pipeline
staging/
├── data/raw/train.csv  # Full dataset
└── .dvc/config  # Points to staging S3 bucket

# Production: Automated runs
production/
├── data/raw/train.csv  # Latest production data
└── .dvc/config  # Points to prod S3 bucket
```

### Workflow

1. **Develop** - Iterate on sample data
1. **Test** - Run on full dataset in staging
1. **Deploy** - Promote to production after validation

______________________________________________________________________

## Key Takeaways

1. **Validation gates prevent bad data from propagating**

    - Fail fast, fail safely
    - Alert humans when intervention needed

1. **Automation enables hands-off operation**

    - Scheduled runs work reliably
    - Human intervention only for exceptions

1. **Versioning enables easy rollback**

    - DVC + Git track exact data state
    - Return to any previous version quickly

1. **Smart caching speeds up iteration**

    - DVC only reruns changed stages
    - Developers iterate faster

1. **Quality over speed**

    - Better to run on slightly outdated data than broken data
    - Reliability > freshness

______________________________________________________________________

## Next Steps

- **[Best Practices](best-practices.md)** - Production-grade recommendations
- **[Validation Guide](validation.md)** - Customize expectations for your use case
- **[Automation](automation.md)** - Set up scheduled pipelines
