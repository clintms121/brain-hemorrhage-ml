"""Prepare the raw dataset: rename the columns, check its quality, and split the list columns.

The raw data has no missing values, duplicates, stray spaces, or impossible
values, so nothing is filled in, trimmed, or removed. check_quality() proves
that on every run, and stops the pipeline if a new version of the data breaks it.
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
# Columns holding several comma-separated values, e.g. "Hypertension, Smoking".
LIST_COLUMNS = ["risk_factors", "symptoms"]
TEXT_COLUMNS = ["patient_id", "gender", "region", "hemorrhage_type", "risk_factors", "symptoms", "outcome"]
MIN_AGE, MAX_AGE = 0, 120


def clean_data(raw, verbose=True):
    """Rename the columns, confirm there's nothing to clean, and split the list columns."""
    df = rename_columns(raw)
    check_quality(df, verbose=verbose)
    return split_list_columns(df)


def rename_columns(df):
    """Switch to short snake_case names, e.g. "Patient ID" -> "patient_id"."""
    return df.rename(columns=COLUMN_NAMES)


def check_quality(df, verbose=True):
    """Count common data problems, and stop if any are found.

    This doesn't change the data. It's the evidence that no cleaning step
    (filling, trimming, or dropping rows) is needed for this dataset.
    """
    problems = find_quality_problems(df)
    if verbose:
        print("Data quality checks:")
        for name, count in problems.items():
            print(f"  {name:<36} {count}")

    found = []
    for name, count in problems.items():
        if count > 0:
            found.append(f"{name}: {count}")
    if found:
        raise ValueError("The data has problems this pipeline doesn't fix: " + "; ".join(found))


def find_quality_problems(df):
    """Return {problem: number of cases} for each check."""
    extra_spaces = 0
    for column in TEXT_COLUMNS:
        text = df[column].dropna()
        extra_spaces += (text != text.str.strip()).sum()

    # Text that isn't a number becomes NaN here, so it's counted as out of range.
    ages = pd.to_numeric(df["age"], errors="coerce")
    bad_ages = ~ages.between(MIN_AGE, MAX_AGE)

    # An unreadable date becomes NaT, which never equals the year, so it's counted too.
    dates = pd.to_datetime(df["date"], errors="coerce")
    years = pd.to_numeric(df["year"], errors="coerce")
    bad_dates = dates.dt.year != years

    return {
        "missing values": int(df.isna().sum().sum()),
        "duplicate rows": int(df.duplicated().sum()),
        "duplicate patient IDs": int(df["patient_id"].duplicated().sum()),
        "text with extra spaces": int(extra_spaces),
        f"ages outside {MIN_AGE}-{MAX_AGE}": int(bad_ages.sum()),
        "dates invalid or not matching year": int(bad_dates.sum()),
    }


def split_list_columns(df):
    """Turn "Smoking, Diabetes" into a list: ["Smoking", "Diabetes"]."""
    df = df.copy()
    for column in LIST_COLUMNS:
        df[column] = df[column].map(_split_items)
    return df


def _split_items(text):
    items = []
    for item in text.split(","):
        items.append(item.strip())
    return items
