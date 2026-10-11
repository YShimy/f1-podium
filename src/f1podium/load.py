"""Load the 14 raw Ergast CSVs with explicit dtypes.

- `\\N` (and truly empty fields) are treated as missing; nothing else is.
- Integer columns use nullable `Int64` so missing values stay visible
  instead of silently turning the column into float.
- Text uses the `string` dtype (available in pandas 2 and 3).
- Date columns are parsed with a fixed format and fail loudly if malformed.
- Clock times and lap times stay as text; parsing them is feature work.

Nothing here writes to disk: data/raw/ is read-only.
"""

from pathlib import Path

import pandas as pd

from f1podium.config import NA_VALUES, RAW_DIR, TABLE_NAMES

I = "Int64"
F = "float64"
S = "string"
D = "date"  # parsed after reading with DATE_FORMAT

DATE_FORMAT = "%Y-%m-%d"

# Every column of every table, in file order. The loader checks that the
# CSV header matches exactly, so a changed dataset version fails loudly.
SCHEMAS: dict[str, dict[str, str]] = {
    "circuits": {
        "circuitId": I, "circuitRef": S, "name": S, "location": S,
        "country": S, "lat": F, "lng": F, "alt": I, "url": S,
    },
    "constructor_results": {
        "constructorResultsId": I, "raceId": I, "constructorId": I,
        "points": F, "status": S,
    },
    "constructor_standings": {
        "constructorStandingsId": I, "raceId": I, "constructorId": I,
        "points": F, "position": I, "positionText": S, "wins": I,
    },
    "constructors": {
        "constructorId": I, "constructorRef": S, "name": S,
        "nationality": S, "url": S,
    },
    "driver_standings": {
        "driverStandingsId": I, "raceId": I, "driverId": I, "points": F,
        "position": I, "positionText": S, "wins": I,
    },
    "drivers": {
        "driverId": I, "driverRef": S, "number": I, "code": S,
        "forename": S, "surname": S, "dob": D, "nationality": S, "url": S,
    },
    "lap_times": {
        "raceId": I, "driverId": I, "lap": I, "position": I, "time": S,
        "milliseconds": I,
    },
    "pit_stops": {
        "raceId": I, "driverId": I, "stop": I, "lap": I, "time": S,
        "duration": S, "milliseconds": I,
    },
    "qualifying": {
        "qualifyId": I, "raceId": I, "driverId": I, "constructorId": I,
        "number": I, "position": I, "q1": S, "q2": S, "q3": S,
    },
    "races": {
        "raceId": I, "year": I, "round": I, "circuitId": I, "name": S,
        "date": D, "time": S, "url": S,
        "fp1_date": D, "fp1_time": S, "fp2_date": D, "fp2_time": S,
        "fp3_date": D, "fp3_time": S, "quali_date": D, "quali_time": S,
        "sprint_date": D, "sprint_time": S,
    },
    "results": {
        "resultId": I, "raceId": I, "driverId": I, "constructorId": I,
        "number": I, "grid": I, "position": I, "positionText": S,
        "positionOrder": I, "points": F, "laps": I, "time": S,
        "milliseconds": I, "fastestLap": I, "rank": I, "fastestLapTime": S,
        "fastestLapSpeed": F, "statusId": I,
    },
    "seasons": {"year": I, "url": S},
    "sprint_results": {
        "resultId": I, "raceId": I, "driverId": I, "constructorId": I,
        "number": I, "grid": I, "position": I, "positionText": S,
        "positionOrder": I, "points": F, "laps": I, "time": S,
        "milliseconds": I, "fastestLap": I, "fastestLapTime": S,
        "statusId": I,
    },
    "status": {"statusId": I, "status": S},
}

assert set(SCHEMAS) == set(TABLE_NAMES), "SCHEMAS and TABLE_NAMES disagree"


def load_table(name: str, raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Read one raw CSV with the declared schema."""
    schema = SCHEMAS[name]
    path = Path(raw_dir) / f"{name}.csv"

    header = pd.read_csv(path, nrows=0).columns.tolist()
    if header != list(schema):
        raise ValueError(
            f"{name}.csv columns do not match schema.\n"
            f"  file:   {header}\n  schema: {list(schema)}"
        )

    date_cols = [c for c, t in schema.items() if t == D]
    # Dates are read as text first, then parsed strictly below.
    dtypes = {c: (S if t == D else t) for c, t in schema.items()}

    df = pd.read_csv(
        path,
        dtype=dtypes,
        na_values=NA_VALUES + [""],
        keep_default_na=False,
    )
    for col in date_cols:
        df[col] = pd.to_datetime(df[col], format=DATE_FORMAT, errors="raise")
    return df


def load_all(raw_dir: Path = RAW_DIR) -> dict[str, pd.DataFrame]:
    """Load all 14 tables into a dict keyed by table name."""
    return {name: load_table(name, raw_dir) for name in TABLE_NAMES}
