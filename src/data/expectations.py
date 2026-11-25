"""The data validation rules (Great Expectations suites).

There are three suites, one for each point where the pipeline checks its data:

- ``raw_train``: data/raw/train.csv, the daily sales history, as it arrives
- ``raw_store``: data/raw/store.csv, one row of facts per store, as it arrives
- ``features``: data/processed/train_features.parquet, just before training

Each rule has a ``notes`` entry that says in plain language why it exists.
The notes appear in the terminal when a rule fails and in the Data Docs report.

To add a rule, add one ``suite.add_expectation(...)`` line to the right suite.
The full list of available expectations is at https://greatexpectations.io/expectations/.
"""

import great_expectations as gx
import great_expectations.expectations as gxe

from features.build_features import get_model_features

# The sample data is 100 stores with about 2.5 years of daily rows each
# (91,992 rows). These bounds catch a truncated or duplicated file.
MIN_TRAIN_ROWS = 50_000
MAX_TRAIN_ROWS = 150_000
NUM_STORES = 100

TRAIN_COLUMNS = [
    "Store",
    "DayOfWeek",
    "Date",
    "Sales",
    "Customers",
    "Open",
    "Promo",
    "StateHoliday",
    "SchoolHoliday",
]

STORE_COLUMNS = [
    "Store",
    "StoreType",
    "Assortment",
    "CompetitionDistance",
    "CompetitionOpenSinceMonth",
    "CompetitionOpenSinceYear",
    "Promo2",
    "Promo2SinceWeek",
    "Promo2SinceYear",
    "PromoInterval",
]

STORE_TYPES = ["a", "b", "c", "d"]
ASSORTMENTS = ["a", "b", "c"]
STATE_HOLIDAYS = ["0", "a", "b", "c"]  # 0 = none, a = public, b = Easter, c = Christmas


def raw_train_suite(store_ids: list[int]) -> gx.ExpectationSuite:
    """Rules for the raw daily sales history (train.csv).

    Parameters
    ----------
    store_ids : list[int]
        The stores listed in store.csv. Every sales row must belong to one of them.
    """
    suite = gx.ExpectationSuite(name="raw_train")

    suite.add_expectation(
        gxe.ExpectTableColumnsToMatchOrderedList(
            column_list=TRAIN_COLUMNS,
            notes="The file has the columns the pipeline reads, in the expected order.",
        )
    )
    suite.add_expectation(
        gxe.ExpectTableRowCountToBeBetween(
            min_value=MIN_TRAIN_ROWS,
            max_value=MAX_TRAIN_ROWS,
            notes="About 92,000 rows are expected; far fewer or more means a bad export.",
        )
    )
    for column in ["Store", "Date", "Sales"]:
        suite.add_expectation(
            gxe.ExpectColumnValuesToNotBeNull(
                column=column,
                notes=f"Every row needs a {column}; the model can't learn from it otherwise.",
            )
        )
    suite.add_expectation(
        gxe.ExpectColumnValuesToBeInSet(
            column="Store",
            value_set=store_ids,
            notes="Every sale belongs to a store listed in store.csv.",
        )
    )
    suite.add_expectation(
        gxe.ExpectColumnValuesToMatchStrftimeFormat(
            column="Date",
            strftime_format="%Y-%m-%d",
            notes="Dates are written as YYYY-MM-DD, such as 2015-07-31.",
        )
    )
    suite.add_expectation(
        gxe.ExpectColumnValuesToBeBetween(
            column="Sales",
            min_value=0,
            max_value=50_000,
            notes="Daily sales are never negative and no store sells over 50,000 a day.",
        )
    )
    suite.add_expectation(
        gxe.ExpectColumnValuesToBeBetween(
            column="Customers",
            min_value=0,
            notes="Customer counts are never negative.",
        )
    )
    suite.add_expectation(
        gxe.ExpectColumnValuesToBeInSet(
            column="DayOfWeek",
            value_set=[1, 2, 3, 4, 5, 6, 7],
            notes="Days of the week are numbered 1 (Monday) to 7 (Sunday).",
        )
    )
    for column in ["Open", "Promo", "SchoolHoliday"]:
        suite.add_expectation(
            gxe.ExpectColumnValuesToBeInSet(
                column=column,
                value_set=[0, 1],
                notes=f"{column} is a yes/no flag: 1 or 0.",
            )
        )
    suite.add_expectation(
        gxe.ExpectColumnValuesToBeInSet(
            column="StateHoliday",
            value_set=STATE_HOLIDAYS,
            notes="StateHoliday is 0 (none), a (public), b (Easter), or c (Christmas).",
        )
    )
    return suite


def raw_store_suite() -> gx.ExpectationSuite:
    """Rules for the raw store facts (store.csv)."""
    suite = gx.ExpectationSuite(name="raw_store")

    suite.add_expectation(
        gxe.ExpectTableColumnsToMatchOrderedList(
            column_list=STORE_COLUMNS,
            notes="The file has the columns the pipeline reads, in the expected order.",
        )
    )
    suite.add_expectation(
        gxe.ExpectTableRowCountToEqual(
            value=NUM_STORES,
            notes="This system uses 100 of Rossmann's 1,115 stores, one row each.",
        )
    )
    suite.add_expectation(
        gxe.ExpectColumnValuesToNotBeNull(
            column="Store",
            notes="Every row identifies its store.",
        )
    )
    suite.add_expectation(
        gxe.ExpectColumnValuesToBeUnique(
            column="Store",
            notes="Each store appears once, so joining it to sales can't duplicate rows.",
        )
    )
    suite.add_expectation(
        gxe.ExpectColumnValuesToBeInSet(
            column="StoreType",
            value_set=STORE_TYPES,
            notes="Rossmann has four store formats, a to d.",
        )
    )
    suite.add_expectation(
        gxe.ExpectColumnValuesToBeInSet(
            column="Assortment",
            value_set=ASSORTMENTS,
            notes="Assortment is a (basic), b (extra), or c (extended).",
        )
    )
    suite.add_expectation(
        gxe.ExpectColumnValuesToBeBetween(
            column="CompetitionDistance",
            min_value=0,
            notes="Distance to the nearest competitor is never negative (blank is allowed).",
        )
    )
    suite.add_expectation(
        gxe.ExpectColumnValuesToBeInSet(
            column="Promo2",
            value_set=[0, 1],
            notes="Promo2 is a yes/no flag: 1 or 0.",
        )
    )
    return suite


def features_suite() -> gx.ExpectationSuite:
    """Rules for the feature table, checked just before training.

    Every model feature listed in config/params.yaml must be present and filled in.
    """
    suite = gx.ExpectationSuite(name="features")

    suite.add_expectation(
        gxe.ExpectTableRowCountToBeBetween(
            min_value=MIN_TRAIN_ROWS,
            max_value=MAX_TRAIN_ROWS,
            notes="Building features keeps every row of the raw sales history.",
        )
    )
    for column in ["Sales", *get_model_features()]:
        suite.add_expectation(
            gxe.ExpectColumnToExist(
                column=column,
                notes=f"The model needs {column}.",
            )
        )
        suite.add_expectation(
            gxe.ExpectColumnValuesToNotBeNull(
                column=column,
                notes=f"{column} is filled in on every row; the model can't use blanks.",
            )
        )
    suite.add_expectation(
        gxe.ExpectColumnValuesToBeBetween(
            column="Sales",
            min_value=0,
            notes="Sales are never negative.",
        )
    )
    suite.add_expectation(
        gxe.ExpectColumnValuesToBeBetween(
            column="Year",
            min_value=2013,
            max_value=2015,
            notes="The sales history runs from January 2013 to July 2015.",
        )
    )
    suite.add_expectation(
        gxe.ExpectColumnValuesToBeBetween(
            column="Month",
            min_value=1,
            max_value=12,
            notes="Months are numbered 1 to 12.",
        )
    )
    suite.add_expectation(
        gxe.ExpectColumnValuesToBeBetween(
            column="Week",
            min_value=1,
            max_value=53,
            notes="Weeks of the year are numbered 1 to 53.",
        )
    )
    suite.add_expectation(
        gxe.ExpectColumnValuesToBeBetween(
            column="CompetitionDistance",
            min_value=0,
            max_value=100_000,
            notes="Missing distances are filled with 100,000 (no nearby competitor).",
        )
    )
    for column in ["Promo", "Promo2", "Promo2Active", "SchoolHoliday"]:
        suite.add_expectation(
            gxe.ExpectColumnValuesToBeInSet(
                column=column,
                value_set=[0, 1],
                notes=f"{column} is a yes/no flag: 1 or 0.",
            )
        )
    for column, values in [
        ("StoreType", STORE_TYPES),
        ("Assortment", ASSORTMENTS),
        ("StateHoliday", STATE_HOLIDAYS),
    ]:
        suite.add_expectation(
            gxe.ExpectColumnValuesToBeInSet(
                column=column,
                value_set=values,
                notes=f"{column} only takes the values seen in the raw data: {values}.",
            )
        )
    return suite
