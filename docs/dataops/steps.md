# Individual DataOps Steps

Each step in the workflow serves a specific purpose and can be run independently for debugging, testing, or incremental updates.

______________________________________________________________________

## Step 1: Validate Raw Data

**Purpose:** Ensure incoming data meets quality expectations **before** investing time in processing. This early validation catches issues like:

- Schema changes (missing columns, wrong data types)
- Value range violations (negative sales, invalid dates)
- Data corruption or formatting errors
- Unexpected nulls in critical fields

**Why This Matters:** Processing bad data wastes compute resources and can introduce subtle bugs that propagate through the entire pipeline. Failing fast at the validation gate saves time and prevents downstream issues.

### Run This Step

```bash
python src/data/validate_data.py --stage raw
```

### What It Validates

- ✅ Expected columns present, in order (`Store`, `Date`, `Sales`, etc.)
- ✅ Value ranges valid (sales between 0 and 50,000, days of the week 1–7)
- ✅ Row counts as expected (100 stores; 50,000–150,000 daily sales rows)
- ✅ Every sale belongs to a store in store.csv, and each store appears once
- ✅ Critical fields non-null; dates written as YYYY-MM-DD

The rules are in `src/data/expectations.py`; see [Data Validation](validation.md) for the full list.

### Success Output

```
raw_store  PASSED  (8 of 8 rules passed)
raw_train  PASSED  (14 of 14 rules passed)

Data Docs: gx/uncommitted/data_docs/local_site/index.html

✓ All data checks passed.
```

### Failure Handling

If validation fails, the script lists each failed rule and why it exists, then exits with code 1, preventing the pipeline from continuing. This is intentional—**bad data should stop the pipeline**, not be silently processed.

______________________________________________________________________

## Step 2: Process Raw Data

**Purpose:** Transform raw CSV files into a clean, merged, analysis-ready dataset. This step performs:

- **Merging**: Combining `train.csv` + `store.csv` on the `Store` key
- **Type Conversion**: Parsing dates, converting categoricals, optimizing dtypes
- **Missing Value Handling**: Filling nulls with business-appropriate defaults
- **Format Optimization**: Saving as Parquet (compressed, columnar format)

**Why This Matters:** Raw data is rarely ready for analysis or modeling. Consistent processing ensures that every data scientist working with this data sees the same clean version, eliminating "works on my machine" issues.

### Run This Step

```bash
python -m src.data.make_dataset
```

### What It Does

1. Loads `data/raw/train.csv` (91,992 rows of daily sales)
1. Loads `data/raw/store.csv` (100 store metadata records)
1. Merges on `Store` column (left join preserves all sales records)
1. Converts `Date` to datetime, `StateHoliday` to category, etc.
1. Fills missing `CompetitionDistance` with median
1. Fills missing `Promo2` metadata with zeros
1. Saves to `data/processed/train_clean.parquet`

### Output File

`data/processed/train_clean.parquet` (~30 MB, compressed)

### Verify Success

```bash
# Check file exists and size
ls -lh data/processed/train_clean.parquet

# Quick inspection
python -c "
import pandas as pd
df = pd.read_parquet('data/processed/train_clean.parquet')
print(f'✓ Rows: {len(df):,}')
print(f'✓ Columns: {len(df.columns)}')
print(f'✓ Date range: {df.Date.min()} to {df.Date.max()}')
print(f'✓ Stores: {df.Store.nunique()}')
"
```

______________________________________________________________________

## Step 3: Build Standard Features

**Purpose:** Create **proven, standard features** that are always useful for Rossmann forecasting models. These are features that have been validated across multiple modeling efforts and should be consistently available.

**Created features:**

- **Calendar features**: Year, Month, Week (of year)
- **Promotion features**: Promo2Active (the store's coupon program runs this month)

The model uses 12 features in total (listed in `config/params.yaml`); the other eight come straight from the raw data. None is computed from past sales, so the API can build them for any future date.

**Why This Matters:** Separating standard features (DataOps) from experimental features (ModelOps) creates a clear boundary:

- **DataOps**: Automated, tested, always created
- **ModelOps**: Experimental, model-specific, optional

This separation enables **faster experimentation** (modelers don't wait for data engineers) and **better reliability** (standard features are thoroughly tested).

### Run This Step

```bash
python -m src.features.build_features
```

### What It Does

1. Loads `data/processed/train_clean.parquet`
1. Adds calendar features (Year, Month, Week)
1. Adds the promotion feature Promo2Active
1. Saves to `data/processed/train_features.parquet`

### Output File

`data/processed/train_features.parquet` (4 new columns)

### Critical Detail

Features must be known before the day being predicted. A feature computed from that day's sales would leak the answer into the model; `tests/test_features.py` checks that no model feature is derived from Sales or Customers.

______________________________________________________________________

## Step 4: Validate Features

**Purpose:** Ensure feature engineering didn't introduce bugs, nulls, or invalid values.

**Why This Matters:** Feature engineering transforms data (date parsing, merges, derived flags). Validation catches errors like:

- Missing columns after a code change
- Null values from a failed merge
- Out-of-range values (for example, a month of 13)

### Run This Step

```bash
python src/data/validate_data.py --stage features
```

The features suite reads the model's feature list from `config/params.yaml` and checks that each feature is present and has no blank values.

______________________________________________________________________

## Step 5: Version Control with DVC

**Purpose:** Track data artifacts (processed data, features) separately from code, enabling:

- **Reproducibility**: Return to exact data state for any model version
- **Collaboration**: Share large datasets without Git bloat
- **Rollback**: Revert to previous data version if issues arise
- **Lineage**: Track which data version produced which model

**Why This Matters:** Git is designed for code, not data. Versioning 100MB+ parquet files in Git causes:

- Repository bloat (slow clones)
- Merge conflicts on binary files
- Poor diff capabilities

DVC solves this by storing **metadata in Git** (small `.dvc` files) and **data in external storage** (local cache or cloud).

### Run This Step

```bash
# Track processed data with DVC
dvc add data/processed/train_clean.parquet
dvc add data/processed/train_features.parquet

# Commit .dvc metadata to Git
git add data/processed/*.dvc .gitignore
git commit -m "data: version processed data and features

- Processed 1,017,209 sales records
- Created 32 standard features
- Validation: PASSED
- Date range: 2013-01-01 to 2015-07-31"
```

### What This Creates

- `train_clean.parquet.dvc`: Metadata file (MD5 hash, file size) tracked in Git
- `train_features.parquet.dvc`: Metadata file tracked in Git
- Actual data files: Moved to `.dvc/cache/` (excluded from Git)
- Updated `.gitignore`: Ensures data files aren't committed to Git

### Optional: Push to Remote

```bash
# Configure remote (one-time, if using cloud storage)
dvc remote add -d myremote s3://my-bucket/rossmann-data

# Push data to remote
dvc push
```

______________________________________________________________________

## Complete Pipeline Diagram

The diagram below shows how **data assets**, **Python scripts**, **validation suites**, and **DVC versioning** work together across Steps 1-5:

```mermaid
flowchart TB
    subgraph Step1["📍 Step 1: Validate Raw Data"]
        direction LR
        R1[📁 data/raw/<br/>train.csv + store.csv]
        GX1[📋 GX suites:<br/>raw_train, raw_store]
        V1[🐍 validate_data.py<br/>--stage raw]
        R1 --> V1
        GX1 -.expectations.-> V1
    end

    V1 -->|✅ PASS| Step2
    V1 -->|❌ FAIL| Stop1[❌ STOP PIPELINE]

    subgraph Step2["📍 Step 2: Process Data"]
        direction LR
        P1[🐍 make_dataset.py<br/>merge + clean]
        OUT1[📁 data/processed/<br/>train_clean.parquet]
        P1 --> OUT1
    end

    OUT1 --> Step3

    subgraph Step3["📍 Step 3: Build Features"]
        direction LR
        F1[🐍 build_features.py<br/>calendar + promo]
        OUT2[📁 data/processed/<br/>train_features.parquet]
        F1 --> OUT2
    end

    OUT2 --> Step4

    subgraph Step4["📍 Step 4: Validate Features"]
        direction LR
        GX3[📋 GX suite:<br/>features]
        V3[🐍 validate_data.py<br/>--stage features]
        GX3 -.expectations.-> V3
    end

    V3 -->|✅ PASS| Step5
    V3 -->|❌ FAIL| Stop3[❌ STOP PIPELINE]

    subgraph Step5["📍 Step 5: Version with DVC"]
        direction LR
        DVC1[⚙️ dvc add<br/>both parquet files]
        META1[📄 *.dvc metadata<br/>tracked in Git]
        CACHE1[💾 .dvc/cache/<br/>data storage]
        DVC1 --> META1
        DVC1 --> CACHE1
    end

    Step5 --> Success[✅ Pipeline Complete<br/>Ready for ModelOps]

    style Stop1 fill:#ffebee,stroke:#c62828,stroke-width:3px
    style Stop3 fill:#ffebee,stroke:#c62828,stroke-width:3px
    style R1 fill:#e3f2fd
    style OUT1 fill:#e8f5e9
    style OUT2 fill:#e8f5e9
    style GX1 fill:#f3e5f5
    style GX3 fill:#f3e5f5
    style META1 fill:#fce4ec
    style CACHE1 fill:#f1f8e9
    style Success fill:#c8e6c9,stroke:#2e7d32,stroke-width:3px
    style Step1 fill:#f5f5f5
    style Step2 fill:#f5f5f5
    style Step3 fill:#f5f5f5
    style Step4 fill:#f5f5f5
    style Step5 fill:#f5f5f5
```

**How the Pipeline Flows:**

1. **Step 1**: Raw CSV files are validated against two Great Expectations suites → PASS/FAIL gate
1. **Step 2**: Processing script reads raw files, merges, cleans, outputs parquet
1. **Step 3**: Feature engineering script reads processed data, creates features
1. **Step 4**: Features validated against a GX suite → PASS/FAIL gate
1. **Step 5**: DVC versions both processed files, creates `.dvc` metadata (Git) and caches data

**Legend:**

- 📁 **Data files** (CSV/Parquet)
- 🐍 **Python scripts** (executable code)
- 📋 **Great Expectations suites** (validation rules)
- 📄 **DVC metadata** (tracked in Git)
- 💾 **DVC cache** (actual data storage)
- Solid arrows (→) = main data flow
- Dotted arrows (-.->)= supporting relationships

______________________________________________________________________

## Next Steps

- **[Data Validation](validation.md)** - Learn about Great Expectations and validation suites
- **[Automation](automation.md)** - Orchestrate with Bash scripts and DVC pipelines
- **[Real-World Scenarios](scenarios.md)** - See how to handle new data arrivals
