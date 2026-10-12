"""Central project configuration: paths, split years, random seed.

Everything that might change between machines or experiments lives here,
so it only has to be edited in one place.
"""

import os
from pathlib import Path

# Repo root, derived from this file's location (src/f1podium/config.py -> repo).
PROJECT_ROOT = Path(__file__).resolve().parents[2]


KAGGLE_INPUT = Path("/kaggle/input")


def find_raw_dir(kaggle_input: Path = KAGGLE_INPUT, default: Path | None = None) -> Path:
    """Locate the folder holding the raw CSVs.

    Order:
      1. F1_DATA_DIR environment variable (explicit override).
      2. If the Kaggle input folder exists: the first folder under it that contains
         races.csv (sorted, so the choice is deterministic). Raises if there is none,
         so a missing dataset fails here with a clear message.
      3. Otherwise data/raw in the repo.
    """
    env_dir = os.environ.get("F1_DATA_DIR")
    if env_dir:
        return Path(env_dir)

    if kaggle_input.exists():
        for races_csv in sorted(kaggle_input.rglob("races.csv")):
            return races_csv.parent
        raise FileNotFoundError(
            f"{kaggle_input} exists but no races.csv was found under it. "
            "Add the Formula 1 dataset to this notebook's inputs."
        )

    return default if default is not None else PROJECT_ROOT / "data" / "raw"


RAW_DIR = find_raw_dir()
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
