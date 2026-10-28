# dev file

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from data.cleaning.print_helpers import *
from data.cleaning.process import run_pipeline

clean, processed = run_pipeline()

print_overview(clean)
print_overview(processed)
