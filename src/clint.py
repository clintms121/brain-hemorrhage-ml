#dev file

from data.cleaning.print_helpers import *
from data.cleaning.process import run_pipeline

clean, processed = run_pipeline()

#print_overview(clean)
#print_overview(processed)

# print first 5 rows
print_rows(clean, 0, 5)

# print dataset dimensions
print("== Dataset dimensions ==")
print(f"Clean:     {clean.shape[0]} rows x {clean.shape[1]} columns")
print(f"Processed: {processed.shape[0]} rows x {processed.shape[1]} columns")
print()

# print data types for the columns used in the project
project_columns = [
    "age",
    "gender",
    "region",
    "year",
    "hemorrhage_type",
    "risk_factors",
    "symptoms",
    "outcome",
]
print("== Data types (columns used in the project) ==")
print(clean[project_columns].dtypes.to_string())