"""Cleaning steps. Each function returns new DataFrames; inputs are never modified.

Pipeline order for `results`:
    exclude_indy500 (D2) -> filter_non_starters (D3) -> resolve_duplicate_results (D1)
"""

import pandas as pd

from f1podium.validate import find_duplicates

KEY = ["raceId", "driverId"]

# D2: the Indianapolis 500 counted for the championship in 1950-1960.
INDY500_CIRCUIT_REF = "indianapolis"
INDY500_YEARS = (1950, 1960)  # inclusive

# D3: statuses that mean "did not start", decided by qualifying (known at the cutoff).
NON_START_STATUSES = ("Did not qualify", "Did not prequalify", "107% Rule")  # regardless of grid
WITHDREW_STATUS = "Withdrew"  # a non-start only when grid == 0 (never took a grid slot)


def _check_result_ids(results: pd.DataFrame, required: set[str]) -> None:
    missing = required - set(results.columns)
    if missing:
        raise ValueError(f"results is missing columns: {sorted(missing)}")
    if results["resultId"].isna().any() or results["resultId"].duplicated().any():
        raise ValueError("resultId must be non-null and unique")


def exclude_indy500(
    results: pd.DataFrame, races: pd.DataFrame, circuits: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Exclude results of the Indianapolis 500 (D2).

    A race is the Indy 500 when it was held at the circuit with circuitRef
    "indianapolis" in 1950-1960. The year range matters: the same circuit
    hosted the United States Grand Prix in 2000-2007, which is kept.

    Returns (kept results, log with one row per excluded race).
    """
    _check_result_ids(results, {"resultId", "raceId"})
    indy_circuits = circuits.loc[circuits["circuitRef"] == INDY500_CIRCUIT_REF, "circuitId"]
    if indy_circuits.empty:
        raise ValueError(f"no circuit with circuitRef {INDY500_CIRCUIT_REF!r}")

    first, last = INDY500_YEARS
    indy_races = races[races["circuitId"].isin(indy_circuits) & races["year"].between(first, last)]

    excluded = results["raceId"].isin(indy_races["raceId"])
    rows_per_race = results.loc[excluded, "raceId"].value_counts()
    log = (
        indy_races[["raceId", "year", "name"]]
        .assign(
            circuitRef=INDY500_CIRCUIT_REF,
            n_rows_excluded=lambda d: d["raceId"].map(rows_per_race).fillna(0).astype(int),
        )
        .sort_values(["year", "raceId"], ignore_index=True)
    )
    return results.loc[~excluded].copy(), log


def filter_non_starters(
    results: pd.DataFrame, status: pd.DataFrame, races: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Exclude entries that did not start the race (D3).

    A row is excluded when its status is one of NON_START_STATUSES (whatever
    its grid), or when its status is "Withdrew" and grid == 0.

    Only the status name and grid are used to select rows. The status values
    used here are settled by qualifying, so they are known at our prediction
    cutoff. Post-race columns such as laps are never used. Deliberately kept:
    "Not classified" (the car started), pit-lane starts (grid == 0 with any
    other status), and "Withdrew" with grid > 0.

    Returns (kept results, log with one row per excluded result).
    """
    _check_result_ids(results, {"resultId", "raceId", "driverId", "statusId", "grid"})
    needed = set(NON_START_STATUSES) | {WITHDREW_STATUS}
    missing = needed - set(status["status"])
    if missing:
        raise ValueError(f"status table is missing: {sorted(missing)}")

    names = results[["statusId"]].merge(status, on="statusId", how="left", validate="many_to_one")["status"]
    if names.isna().any():
        raise ValueError("some results have a statusId that is not in the status table")
    names.index = results.index

    by_status = names.isin(NON_START_STATUSES)
    withdrew_no_grid = (names == WITHDREW_STATUS) & results["grid"].eq(0).fillna(False)
    excluded = by_status | withdrew_no_grid

    log = results.loc[excluded, ["resultId", "raceId", "driverId", "grid"]].assign(
        status=names[excluded],
        reason=pd.Series("status: ", index=names[excluded].index) + names[excluded],
    )
    log.loc[withdrew_no_grid[excluded], "reason"] = "Withdrew with grid = 0"
    log = log.merge(races[["raceId", "year"]], on="raceId", how="left", validate="many_to_one")
    if log["year"].isna().any():
        raise ValueError("some excluded rows have a raceId that is not in races")
    log = log[["resultId", "raceId", "driverId", "year", "status", "grid", "reason"]].sort_values(
        ["year", "raceId", "resultId"], ignore_index=True
    )
    return results.loc[~excluded].copy(), log

# Values of `resolving_step` in the duplicate log.
STEP_NON_STARTER = "non_starter_filter"    # no row started -> left for the non-starter filter
STEP_STARTED_ROW = "single_started_row"    # rule 2: exactly one row with grid > 0
STEP_LOWEST_GRID = "lowest_grid"           # rule 3: several started rows, unique lowest grid
STEP_LOWEST_RESULT_ID = "lowest_resultId"  # rule 4: tie on lowest grid

LOG_COLUMNS = [
    "raceId", "driverId", "year", "n_rows", "resolving_step",
    "kept_resultId", "dropped_resultIds", "lowest_grid_differs_from_lowest_resultId",
]


def resolve_duplicate_results(
    results: pd.DataFrame, races: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Resolve duplicate (raceId, driverId) rows in `results` (rule B).

    Rule B is a target-blind heuristic for the driver's primary entry. It is NOT
    "keep the car the driver started in"; the data cannot identify that. Every
    resolved pair in the real data has more than one row with grid > 0, and each
    row carries *that car's* grid slot, so the lowest grid picks the better-placed
    car the driver is listed with. What it guarantees is that the choice never
    looks at the outcome (positions, points, status) and is deterministic.

    For each duplicated pair:
      1. If no row has grid > 0, every row is a non-starter: leave the rows in place
         and log the pair as resolved by the (separate, later) non-starter filter.
      2. If exactly one row has grid > 0, keep it.
      3. If several rows have grid > 0, keep the lowest grid.
      4. If the lowest grid is tied, keep the lowest resultId.

    "Started" means grid > 0; a null grid counts as not started. This is safe for
    the duplicate pairs in this dataset (all 1978 or earlier); in modern seasons
    grid = 0 can also mean a pit-lane start.

    Known misattributions: rule 3 can keep a car the driver took over rather than
    the one they started in, e.g. Fagioli 1951 (raceId 828) and Brooks 1957
    (raceId 780). The log flags pairs where lowest grid and lowest resultId disagree.

    Uses only grid (pre-race) and resultId (a row id): no positions, points or status.

    Returns (cleaned results, log with one row per duplicated pair).
    """
    _check_result_ids(results, {"resultId", "raceId", "driverId", "grid"})

    work = results.copy()
    dups = find_duplicates(work, KEY)

    log_rows = []
    all_dropped: list[int] = []
    for (race_id, driver_id), rows in dups.groupby(KEY, sort=True):
        started = rows[rows["grid"].fillna(0) > 0]

        if started.empty:
            step, kept, differs = STEP_NON_STARTER, pd.NA, pd.NA
        else:
            if len(started) == 1:
                step = STEP_STARTED_ROW
            elif (started["grid"] == started["grid"].min()).sum() == 1:
                step = STEP_LOWEST_GRID
            else:
                step = STEP_LOWEST_RESULT_ID
            kept = int(started.sort_values(["grid", "resultId"])["resultId"].iloc[0])
            differs = kept != int(started["resultId"].min())

        dropped = [] if pd.isna(kept) else sorted(int(r) for r in rows["resultId"] if r != kept)
        all_dropped.extend(dropped)
        log_rows.append({
            "raceId": race_id,
            "driverId": driver_id,
            "n_rows": len(rows),
            "resolving_step": step,
            "kept_resultId": kept,
            "dropped_resultIds": dropped,
            "lowest_grid_differs_from_lowest_resultId": differs,
        })

    log = pd.DataFrame(log_rows, columns=[c for c in LOG_COLUMNS if c != "year"])
    log = log.merge(races[["raceId", "year"]], on="raceId", how="left", validate="many_to_one")
    if log["year"].isna().any():
        raise ValueError("some duplicated pairs have a raceId that is not in races")
    log = (
        log.astype({"kept_resultId": "Int64", "lowest_grid_differs_from_lowest_resultId": "boolean"})
        [LOG_COLUMNS]
        .sort_values(["year", "raceId", "driverId"], ignore_index=True)
    )

    cleaned = work[~work["resultId"].isin(all_dropped)]
    return cleaned, log


# Values of `step` in the combined clean_results log, in pipeline order.
CLEAN_STEP_INDY500 = "D2: Indianapolis 500"
CLEAN_STEP_NON_STARTER = "D3: non-starter"
CLEAN_STEP_DUPLICATE = "D1: duplicate"

CLEAN_LOG_COLUMNS = [
    "step", "resultId", "raceId", "driverId", "year", "reason",
    # D1 only (pair-level details repeated on each dropped row; <NA> for D2/D3):
    "resolving_step", "kept_resultId", "lowest_grid_differs_from_lowest_resultId",
]


def clean_results(
    results: pd.DataFrame,
    races: pd.DataFrame,
    circuits: pd.DataFrame,
    status: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the results cleaning pipeline in its fixed order.

        exclude_indy500 (D2) -> filter_non_starters (D3) -> resolve_duplicate_results (D1)

    The order matters: D3 must run before D1 so that a non-starter row can never
    be the row D1 keeps for a duplicated pair, and D2 runs first so that rows of
    excluded races are attributed to D2 rather than to a later step.

    Returns (cleaned results, combined log). The log has one row per removed
    results row, labelled with the step that removed it.

    Raises AssertionError if the cleaned table is not unique on (raceId, driverId)
    or if the row counts do not add up.
    """
    no_indy, indy_log = exclude_indy500(results, races, circuits)
    started, non_start_log = filter_non_starters(no_indy, status, races)
    cleaned, dup_log = resolve_duplicate_results(started, races)

    d2 = (
        results.loc[results["raceId"].isin(indy_log["raceId"]), ["resultId", "raceId", "driverId"]]
        .merge(indy_log[["raceId", "year", "name"]], on="raceId", how="left", validate="many_to_one")
        .assign(step=CLEAN_STEP_INDY500, reason=lambda d: d["name"] + " (excluded race)")
    )
    d3 = non_start_log.assign(step=CLEAN_STEP_NON_STARTER)
    d1 = (
        dup_log.explode("dropped_resultIds")
        .dropna(subset=["dropped_resultIds"])
        .rename(columns={"dropped_resultIds": "resultId"})
        .assign(
            step=CLEAN_STEP_DUPLICATE,
            reason=lambda d: "duplicate pair: kept resultId " + d["kept_resultId"].astype(str),
        )
    )
    # Skip empty parts: concatenating empty frames triggers a pandas FutureWarning about dtypes.
    parts = [part.reindex(columns=CLEAN_LOG_COLUMNS) for part in (d2, d3, d1) if len(part)]
    log = pd.concat(
        parts or [pd.DataFrame(columns=CLEAN_LOG_COLUMNS)], ignore_index=True
    ).astype({
        "step": "string", "resultId": "Int64", "raceId": "Int64", "driverId": "Int64", "year": "Int64",
        "reason": "string", "resolving_step": "string", "kept_resultId": "Int64",
        "lowest_grid_differs_from_lowest_resultId": "boolean",
    })

    if log["resultId"].duplicated().any():
        raise AssertionError("a results row was removed by more than one step")
    if len(cleaned) != len(results) - len(log):
        raise AssertionError("row counts do not add up: cleaned != results - removed")
    if cleaned.duplicated(KEY).any():
        raise AssertionError("(raceId, driverId) is not unique after cleaning")
    return cleaned, log
