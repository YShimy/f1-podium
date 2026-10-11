"""Central project configuration: paths, split years, random seed.

Everything that might change between machines or experiments lives here,
so it only has to be edited in one place.
"""

import os
from pathlib import Path

# Repo root, derived from this file's location (src/f1podium/config.py -> repo).
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _find_raw_dir() -> Path:
    """Locate the raw CSV folder.

    Order: F1_DATA_DIR env var -> Kaggle input dataset -> data/raw in the repo.
    """
    env_dir = os.environ.get("F1_DATA_DIR")
    if env_dir:
        return Path(env_dir)

    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        for races_csv in sorted(kaggle_input.rglob("races.csv")):
            return races_csv.parent

    return PROJECT_ROOT / "data" / "raw"


RAW_DIR = _find_raw_dir()
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# Chronological split by races.year.
TRAIN_END_YEAR = 2019
VAL_YEARS = (2020, 2021)
TEST_START_YEAR = 2022

RANDOM_SEED = 42

# Missing-value sentinel used throughout the Ergast CSVs.
NA_VALUES = ["\\N"]

TABLE_NAMES = [
    "circuits",
    "constructor_results",
    "constructor_standings",
    "constructors",
    "driver_standings",
    "drivers",
    "lap_times",
    "pit_stops",
    "qualifying",
    "races",
    "results",
    "seasons",
    "sprint_results",
    "status",
]
