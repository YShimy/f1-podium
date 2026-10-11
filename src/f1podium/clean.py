"""Cleaning steps. Each function returns new DataFrames; inputs are never modified."""

import pandas as pd

from f1podium.validate import find_duplicates

KEY = ["raceId", "driverId"]

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
    missing = {"resultId", "raceId", "driverId", "grid"} - set(results.columns)
    if missing:
        raise ValueError(f"results is missing columns: {sorted(missing)}")
    if results["resultId"].isna().any() or results["resultId"].duplicated().any():
        raise ValueError("resultId must be non-null and unique (rows are dropped by resultId)")

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
