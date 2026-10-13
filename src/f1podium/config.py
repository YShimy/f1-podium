"""Central project configuration: paths, split years, random seed.

Everything that might change between machines or experiments lives here,
so it only has to be edited in one place.
"""

import os
from pathlib import Path

# Repo root, derived from this file's location (src/f1podium/config.py -> repo).
PROJECT_ROOT = Path(__file__).resolve().parents[2]


KAGGLE_INPUT = Path("/kaggle/input")
KAGGLE_MAX_DEPTH = 6   # Kaggle mounts at /kaggle/input/datasets/<owner>/<slug>/..., so search deeply
ERROR_TREE_DEPTH = 4


def describe_tree(root: Path, max_depth: int = ERROR_TREE_DEPTH) -> str:
    """Indented listing of root (directories and files) down to max_depth, for error messages."""
    lines = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        here = Path(dirpath)
        depth = len(here.relative_to(root).parts)
        lines.append("  " * depth + (here.name or str(root)) + "/")
        lines.extend("  " * (depth + 1) + name for name in sorted(filenames))
        if depth >= max_depth:
            dirnames[:] = []
    return "\n".join(lines)


def find_folder_with_file(root: Path, filename: str, max_depth: int = KAGGLE_MAX_DEPTH) -> Path | None:
    """First folder (sorted walk, at most max_depth levels below root) containing filename."""
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        here = Path(dirpath)
        if filename in filenames:
            return here
        if len(here.relative_to(root).parts) >= max_depth:
            dirnames[:] = []
    return None


def find_raw_dir(kaggle_input: Path = KAGGLE_INPUT, default: Path | None = None) -> Path:
    """Locate the folder holding the raw CSVs.

    Order:
      1. F1_DATA_DIR environment variable (explicit override).
      2. If the Kaggle input folder exists: the first folder under it (up to 6 levels deep,
         sorted, so the choice is deterministic) that contains races.csv. Raises, listing
         the folder tree, if there is none, rather than falling back to the repo.
      3. Otherwise data/raw in the repo.
    """
    env_dir = os.environ.get("F1_DATA_DIR")
    if env_dir:
        return Path(env_dir)

    if kaggle_input.exists():
        found = find_folder_with_file(kaggle_input, "races.csv")
        if found is not None:
            return found
        raise FileNotFoundError(
            f"{kaggle_input} exists but no races.csv was found under it (searched {KAGGLE_MAX_DEPTH} levels). "
            "Add the Formula 1 dataset to this notebook's inputs. Contents "
            f"(depth <= {ERROR_TREE_DEPTH}):\n{describe_tree(kaggle_input)}"
        )

    return default if default is not None else PROJECT_ROOT / "data" / "raw"


RAW_DIR = find_raw_dir()
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def find_results_dir(kaggle_working: Path = Path("/kaggle/working")) -> Path:
    """Where evaluation results are written.

    Order: F1_RESULTS_DIR environment variable; /kaggle/working/results on Kaggle (the input
    folders there are read-only); otherwise results/ in the repo.
    """
    env_dir = os.environ.get("F1_RESULTS_DIR")
    if env_dir:
        return Path(env_dir)
    if kaggle_working.exists():
        return kaggle_working / "results"
    return PROJECT_ROOT / "results"


RESULTS_DIR = find_results_dir()

# Chronological split by races.year. The model is fitted on TRAIN_START_YEAR..TRAIN_END_YEAR only.
TRAIN_START_YEAR = 1950
TRAIN_END_YEAR = 2019
VAL_YEARS = (2020, 2021)
TEST_START_YEAR = 2022

RANDOM_SEED = 42
BOOTSTRAP_B = 1000                      # race-level bootstrap resamples (seed: RANDOM_SEED)

# Tree-model grids. Small and stated; every value is tried and the best VALIDATION PR-AUC is chosen
# (ties: the first in the order listed). The test split is never used to choose anything.
RF_GRID = {"n_estimators": (200, 500), "max_depth": (6, 12, None), "min_samples_leaf": (1, 5, 20)}
HGB_GRID = {"learning_rate": (0.03, 0.1), "max_iter": (100, 300), "max_depth": (3, 6),
            "l2_regularization": (0.0, 1.0)}
CLASS_WEIGHTS = (None, "balanced")      # compared once per model at its chosen grid configuration

# Data-engineering questions (descriptive analyses of race outcomes; not model features).
DE_WINDOW = (2003, 2024)          # primary window: seasons with near-complete qualifying data
DE_ERA_SPLIT_YEAR = 2014          # eras: <= 2013 vs >= 2014
MIN_FRONT_ROW_STARTS = 10         # minimum front-row starts per circuit-era to report a rate

# Q2 (home-country advantage). Grid buckets as (label, first, last); last=None means "and above".
GRID_BUCKETS = [("1", 1, 1), ("2–3", 2, 3), ("4–6", 4, 6), ("7–10", 7, 10), ("11+", 11, None)]
GRID_BUCKET_PIT_LANE = "0 (pit lane)"   # grid == 0
Q2_MIN_HOME_RACES = 1                   # sensitivity (a): drivers with at least this many home races ...
Q2_MIN_AWAY_RACES = 10                  # ... and at least this many away races
Q2_RECENT_START_YEAR = 2003             # sensitivity (b): seasons from this year on

# Q3 (mechanical retirements per constructor). Eras as (label, first year, last year).
Q3_ERAS = (("2014–2021", 2014, 2021), ("2022–2024", 2022, 2024))
Q3_MIN_STARTS = 40                      # minimum starts per constructor-era (about one season, two cars)

# Modeling table. Columns that are only known after the race: never features for the CURRENT race.
FORBIDDEN_CURRENT_RACE_COLUMNS = (
    "position", "positionText", "positionOrder", "points", "laps", "time", "milliseconds",
    "fastestLap", "rank", "fastestLapTime", "fastestLapSpeed", "statusId", "status",
)
# Tables whose rows for the current raceId must never feed a feature (standings are cumulative AFTER
# the race; the current round's sprint is also excluded).
FORBIDDEN_CURRENT_RACE_TABLES = (
    "lap_times", "pit_stops", "constructor_results", "driver_standings", "constructor_standings",
    "sprint_results",
)
ROLLING_WINDOWS = (3, 5)                # form over the last N prior races
MECH_WINDOW = 10                        # constructor mechanical-retirement rate over the last N prior races
CONSTRUCTOR_KEY = "lineage"             # key for con_* features: "lineage" or "constructorId" (ablation)

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
