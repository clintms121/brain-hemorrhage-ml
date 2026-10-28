"""Clean the raw dataset into consistent, correctly typed columns.

Each step takes a DataFrame and returns a new one, so steps can be run
one at a time while exploring, or all together with clean_data().
"""

import pandas as pd

COLUMN_NAMES = {
    "Patient ID": "patient_id",
    "Age": "age",
    "Gender": "gender",
    "Region": "region",
    "Date": "date",
    "Year": "year",
    "Hemorrhage Type": "hemorrhage_type",
    "Medical History (Risk Factors)": "risk_factors",
    "Symptoms": "symptoms",
    "Outcome": "outcome",
}
CATEGORY_COLUMNS = ["gender", "region", "hemorrhage_type", "outcome"]
# Columns holding several comma-separated values, e.g. "Hypertension, Smoking".
LIST_COLUMNS = ["risk_factors", "symptoms"]
TEXT_COLUMNS = ["patient_id"] + CATEGORY_COLUMNS + LIST_COLUMNS
# A missing date is fine as long as year is present, and a missing
# risk_factors value is filled with "None/Unknown" instead of being dropped.
REQUIRED_COLUMNS = ["patient_id", "age", "year"] + CATEGORY_COLUMNS + ["symptoms"]
UNKNOWN_RISK = "None/Unknown"
MIN_AGE, MAX_AGE = 0, 120


def clean_data(raw, verbose=True):
    """Run every cleaning step in order and return the cleaned DataFrame."""
    df = raw
    steps = [
        rename_columns,
        strip_text,
        remove_duplicates,
        fix_types,
        remove_invalid_rows,
        split_list_columns,
    ]
    for step in steps:
        rows_before = len(df)
        df = step(df)
        if verbose:
            print(f"{step.__name__:<20} {rows_before:>5} -> {len(df):>5} rows")
    return df.reset_index(drop=True)


def rename_columns(df):
    """Switch to short snake_case names, e.g. "Patient ID" -> "patient_id"."""
    return df.rename(columns=COLUMN_NAMES)


def strip_text(df):
    """Trim stray spaces and treat blank strings as missing."""
    df = df.copy()
    for column in TEXT_COLUMNS:
        text = df[column].str.strip()
        df[column] = text.where(text != "")
    return df


def remove_duplicates(df):
    """Drop repeated rows, then keep only the first row for each patient ID."""
    df = df.drop_duplicates()
    return df.drop_duplicates(subset="patient_id", keep="first")


def fix_types(df):
    """Make age and year numeric and date a real datetime.

    Anything that can't be parsed becomes missing, and remove_invalid_rows
    drops it. Year is taken from the date when there is one, so the two
    always agree.
    """
    df = df.copy()
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Use the date's year when there is a date, otherwise keep the year column's value.
    year_from_date = df["date"].dt.year
    year_column = pd.to_numeric(df["year"], errors="coerce")
    df["year"] = year_from_date.fillna(year_column)
    return df


def remove_invalid_rows(df):
    """Drop rows missing a required value or with an impossible age."""
    df = df.dropna(subset=REQUIRED_COLUMNS)
    df = df[df["age"].between(MIN_AGE, MAX_AGE)]
    return df.astype({"age": int, "year": int})


def split_list_columns(df):
    """Turn "Smoking, Diabetes" into a sorted list: ["Diabetes", "Smoking"]."""
    df = df.copy()
    df["risk_factors"] = df["risk_factors"].fillna(UNKNOWN_RISK)
    for column in LIST_COLUMNS:
        df[column] = df[column].map(_split_items)
    return df


def _split_items(text):
    """Split "Smoking, Diabetes" into ["Diabetes", "Smoking"], skipping blanks and repeats."""
    items = []
    for item in text.split(","):
        item = item.strip()
        if item and item not in items:
            items.append(item)
    return sorted(items)
