"""
Run data cleaning pipeline
"""

import re

import pandas as pd

from .clean import LIST_COLUMNS, clean_data
from .import_data import PROCESSED_DATA_PATH, load_raw_data

# Outcomes numbered from best to worst, so a higher code means a worse outcome.
OUTCOME_CODES = {
    "Recovered": 0,
    "Rehabilitation Required": 1,
    "Severe Disability": 2,
    "Fatal": 3,
}
ONE_HOT_COLUMNS = ["gender", "region", "hemorrhage_type"]
LIST_PREFIXES = {"risk_factors": "risk", "symptoms": "symptom"}
ID_COLUMN = "patient_id"
TARGET_COLUMN = "outcome"


def process_data(clean):
    """Encode a cleaned DataFrame as numbers.

    - risk_factors / symptoms -> one 0/1 column per item (risk_smoking, ...)
    - gender / region / hemorrhage_type -> one 0/1 column per category
    - outcome -> 0 (Recovered) to 3 (Fatal), see OUTCOME_CODES
    - date is dropped because year already covers it

    Age is left unscaled on purpose: fit any scaler on the training split
    only, so the test set doesn't leak into it.
    """
    df = clean.drop(columns=["date"])
    for column in LIST_COLUMNS:
        df = encode_list_column(df, column, LIST_PREFIXES[column])
    df = pd.get_dummies(df, columns=ONE_HOT_COLUMNS, dtype=int)
    df = encode_outcome(df)
    df = df.rename(columns=_snake_case)

    # Move the target to the last column so it's easy to spot.
    target = df.pop(TARGET_COLUMN)
    df[TARGET_COLUMN] = target
    return df


def encode_list_column(df, column, prefix):
    """Replace a column of lists with one 0/1 column per possible item."""
    # Join each list into one string, e.g. ["Diabetes", "Smoking"] -> "Diabetes|Smoking",
    # so get_dummies can split it back up and make one 0/1 column per item.
    joined = df[column].str.join("|")
    dummies = joined.str.get_dummies(sep="|")
    dummies = dummies.add_prefix(f"{prefix}_")

    df = df.drop(columns=[column])
    return pd.concat([df, dummies], axis=1)


def encode_outcome(df):
    """Map outcome labels to 0-3 and fail loudly on a label we don't expect."""
    # Labels that aren't in OUTCOME_CODES become NaN.
    codes = df[TARGET_COLUMN].map(OUTCOME_CODES)
    unknown = df.loc[codes.isna(), TARGET_COLUMN].unique()
    if len(unknown) > 0:
        raise ValueError(f"Unexpected outcome values: {list(unknown)}")

    df = df.copy()
    df[TARGET_COLUMN] = codes.astype(int)
    return df


def split_features_target(processed):
    """Return (X, y) for training: every feature column, and the outcome codes."""
    X = processed.drop(columns=[ID_COLUMN, TARGET_COLUMN])
    y = processed[TARGET_COLUMN]
    return X, y


def save_data(df, path=PROCESSED_DATA_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def run_pipeline(verbose=True):
    """Load, clean, process, and save. Returns the cleaned and processed data."""
    raw = load_raw_data()
    clean = clean_data(raw, verbose=verbose)
    processed = process_data(clean)
    path = save_data(processed)
    if verbose:
        print(f"Saved {processed.shape[0]} rows x {processed.shape[1]} columns to {path}")
    return clean, processed


def _snake_case(name):
    """Lowercase with underscores, e.g. "symptom_Stiff Neck" -> "symptom_stiff_neck"."""
    return re.sub(r"[^0-9a-z]+", "_", name.lower()).strip("_")
