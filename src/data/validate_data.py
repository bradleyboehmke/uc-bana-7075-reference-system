"""Validate the pipeline's data with Great Expectations.

The pipeline checks its data at two points:

- ``raw``: the two raw files as they arrive (train.csv and store.csv)
- ``features``: the feature table, just before training

The rules are in src/data/expectations.py. If any rule fails, this script
prints which ones and why, then exits with code 1 so the pipeline stops before
bad data reaches the model. Every run also updates the Data Docs report, a
browsable web page of the results, at gx/uncommitted/data_docs/local_site/index.html.

The gx/ folder is Great Expectations' working folder. This script creates it on
the first run and it is not committed: the rules live in expectations.py.

Usage
-----
    python src/data/validate_data.py --stage raw
    python src/data/validate_data.py --stage features
    python src/data/validate_data.py            # both
"""

import argparse
import sys
import warnings
from pathlib import Path

import great_expectations as gx
import pandas as pd
from great_expectations.data_context.types.base import ProgressBarsConfig

sys.path.insert(0, str(Path(__file__).parents[1]))

from data.expectations import features_suite, raw_store_suite, raw_train_suite  # noqa: E402
from utils.io import PROJECT_ROOT  # noqa: E402

RAW_TRAIN_PATH = PROJECT_ROOT / "data" / "raw" / "train.csv"
RAW_STORE_PATH = PROJECT_ROOT / "data" / "raw" / "store.csv"
FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "train_features.parquet"
DATA_DOCS_PATH = Path("gx") / "uncommitted" / "data_docs" / "local_site" / "index.html"

# GE warns about optional features this project doesn't use.
warnings.filterwarnings("ignore", module="great_expectations")


def get_context(project_root: Path = PROJECT_ROOT):
    """Load the Great Expectations project in ``<project_root>/gx/``, creating it if needed."""
    context = gx.get_context(mode="file", project_root_dir=str(project_root))
    if context.variables.progress_bars is None:
        # Progress bars flood the terminal and notebook output; turn them off once.
        context.variables.progress_bars = ProgressBarsConfig(globally=False)
        context.variables.save()
    return context


def run_checkpoint(context, df: pd.DataFrame, suite: gx.ExpectationSuite):
    """Check a dataframe against a suite, save the result, and update Data Docs.

    This is GE's standard pattern: a data source and batch definition say what
    data to check, a validation definition pairs that data with a suite, and a
    checkpoint runs the validation and then its actions (here, update Data Docs).

    Returns
    -------
    great_expectations.core.ExpectationSuiteValidationResult
        The result; ``result.success`` is True when every rule passed.
    """
    name = suite.name
    data_source = context.data_sources.add_or_update_pandas(name="rossmann")
    batch_definition = data_source.add_dataframe_asset(
        name=name
    ).add_batch_definition_whole_dataframe(name=f"{name}_batch")

    suite = context.suites.add_or_update(suite)
    validation = context.validation_definitions.add_or_update(
        gx.ValidationDefinition(name=f"{name}_validation", data=batch_definition, suite=suite)
    )
    checkpoint = context.checkpoints.add_or_update(
        gx.Checkpoint(
            name=f"{name}_checkpoint",
            validation_definitions=[validation],
            actions=[gx.checkpoint.UpdateDataDocsAction(name="update_data_docs")],
            result_format="SUMMARY",
        )
    )
    checkpoint_result = checkpoint.run(batch_parameters={"dataframe": df})
    return next(iter(checkpoint_result.run_results.values()))


def describe_failures(result) -> list[str]:
    """Return one plain-language line per failed rule."""
    lines = []
    for check in result.results:
        if check.success:
            continue
        config = check.expectation_config
        column = config.kwargs.get("column")
        target = f" on {column}" if column else ""
        line = f"{config.type}{target}"

        details = check.result or {}
        count = details.get("unexpected_count")
        if count:
            examples = details.get("partial_unexpected_list", [])[:5]
            rows = "row" if count == 1 else "rows"
            line += f": {count:,} {rows} failed, for example {examples}"
        elif "observed_value" in details:
            line += f": found {details['observed_value']}"

        note = config.notes
        if note:
            line += f"\n      Rule: {note}"
        lines.append(line)
    return lines


def read_raw_train(path: Path = RAW_TRAIN_PATH) -> pd.DataFrame:
    """Read train.csv. StateHoliday mixes 0 and "0" in the file, so read it as text."""
    return pd.read_csv(path, dtype={"StateHoliday": str})


def validate_raw(context, train_path: Path = RAW_TRAIN_PATH, store_path: Path = RAW_STORE_PATH):
    """Validate store.csv and train.csv. Returns {suite name: result}."""
    store = pd.read_csv(store_path)
    train = read_raw_train(train_path)
    store_ids = sorted(int(s) for s in store["Store"].dropna().unique())
    return {
        "raw_store": run_checkpoint(context, store, raw_store_suite()),
        "raw_train": run_checkpoint(context, train, raw_train_suite(store_ids)),
    }


def validate_features(context, features_path: Path = FEATURES_PATH):
    """Validate the feature table. Returns {suite name: result}."""
    df = pd.read_parquet(features_path)
    return {"features": run_checkpoint(context, df, features_suite())}


def report(results: dict) -> bool:
    """Print a summary of the results. Returns True when every suite passed."""
    print()
    for name, result in results.items():
        stats = result.statistics
        status = "PASSED" if result.success else "FAILED"
        print(
            f"{name:10s} {status}  "
            f"({stats['successful_expectations']} of {stats['evaluated_expectations']} rules passed)"
        )
        for line in describe_failures(result):
            print(f"  ✗ {line}")
    print(f"\nData Docs: {DATA_DOCS_PATH}")
    return all(result.success for result in results.values())


def main(argv: list[str] | None = None) -> int:
    """Run the requested validation stage. Returns the exit code (0 = passed, 1 = failed)."""
    parser = argparse.ArgumentParser(description="Validate data with Great Expectations")
    parser.add_argument(
        "--stage",
        choices=["raw", "features", "all"],
        default="all",
        help="Which data to validate (default: all)",
    )
    args = parser.parse_args(argv)

    context = get_context()
    results = {}
    if args.stage in ("raw", "all"):
        results.update(validate_raw(context, RAW_TRAIN_PATH, RAW_STORE_PATH))
    if args.stage in ("features", "all"):
        results.update(validate_features(context, FEATURES_PATH))

    if report(results):
        print("\n✓ All data checks passed.")
        return 0
    print("\n✗ Data checks failed. Fix the data before running the next pipeline step.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
