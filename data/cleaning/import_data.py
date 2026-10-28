"""Load the raw brain hemorrhage dataset from disk."""

from pathlib import Path

import pandas as pd

# Paths are built from this file's location, so they work from any working directory.
DATA_DIR = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = DATA_DIR / "brain_hemorrhage_dataset.csv"
PROCESSED_DATA_PATH = DATA_DIR / "processed" / "brain_hemorrhage_processed.csv"


def load_raw_data(path=RAW_DATA_PATH):
    """Read the raw CSV exactly as it is on disk."""
    return pd.read_csv(path)
