"""Load, clean, and process the brain hemorrhage dataset.

From the project root:
    from data.cleaning import run_pipeline, print_where
    clean, processed = run_pipeline()
    print_where(clean, outcome="Fatal", n=5)
"""

from .clean import clean_data
from .import_data import load_raw_data
from .print_helpers import (
    print_columns,
    print_counts,
    print_crosstab,
    print_overview,
    print_patient,
    print_rows,
    print_where,
)
from .process import OUTCOME_CODES, process_data, run_pipeline, split_features_target
