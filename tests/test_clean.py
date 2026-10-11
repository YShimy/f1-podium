"""Tests for the results cleaning steps on small fake tables."""

import pandas as pd
import pytest

from f1podium.clean import (
    STEP_LOWEST_GRID,
    STEP_LOWEST_RESULT_ID,
    STEP_NON_STARTER,
    STEP_STARTED_ROW,
    exclude_indy500,
    filter_non_starters,
    resolve_duplicate_results,
)

# raceId 1..4 each hold one duplicated pair for driverId 10; driverId 20 is a normal row.
# In the tie case the higher resultId is listed first, so "keep lowest resultId"
# cannot pass by accident by keeping the first row.
FAKE_ROWS = [
    # resultId, raceId, driverId, grid
    (11, 1, 10, 5),  # one started row -> keep 11
    (12, 1, 10, 0),
    (21, 2, 10, 7),  # two started rows -> keep lowest grid (22)
    (22, 2, 10, 3),
    (41, 3, 10, 4),  # tie on grid -> keep lowest resultId (40)
    (40, 3, 10, 4),
    (51, 4, 10, 0),  # all non-starters -> left in place
    (52, 4, 10, 0),
    (60, 1, 20, 2),  # not duplicated
]


@pytest.fixture
def results():
    df = pd.DataFrame(FAKE_ROWS, columns=["resultId", "raceId", "driverId", "grid"])
    return df.astype("Int64")


@pytest.fixture
def races():
    return pd.DataFrame({"raceId": [1, 2, 3, 4], "year": [1950, 1951, 1952, 1953]}).astype("Int64")


def log_row(log, race_id):
    row = log[log["raceId"] == race_id]
    assert len(row) == 1
    return row.iloc[0]


def test_each_step_keeps_the_expected_row(results, races):
    _, log = resolve_duplicate_results(results, races)
    expected = {
        1: (STEP_STARTED_ROW, 11, [12]),
        2: (STEP_LOWEST_GRID, 22, [21]),
        3: (STEP_LOWEST_RESULT_ID, 40, [41]),
    }
    for race_id, (step, kept, dropped) in expected.items():
        row = log_row(log, race_id)
        assert row["resolving_step"] == step
        assert row["kept_resultId"] == kept
        assert row["dropped_resultIds"] == dropped
        assert row["n_rows"] == 2


def test_all_non_starters_are_left_for_the_non_starter_filter(results, races):
    cleaned, log = resolve_duplicate_results(results, races)
    row = log_row(log, 4)
    assert row["resolving_step"] == STEP_NON_STARTER
    assert pd.isna(row["kept_resultId"])
    assert row["dropped_resultIds"] == []
    assert set(cleaned.loc[cleaned["raceId"] == 4, "resultId"]) == {51, 52}


def test_log_has_one_row_per_pair_with_year(results, races):
    _, log = resolve_duplicate_results(results, races)
    assert len(log) == 4
    assert log["year"].tolist() == [1950, 1951, 1952, 1953]


def test_row_count_and_uniqueness(results, races):
    cleaned, log = resolve_duplicate_results(results, races)
    n_dropped = log["dropped_resultIds"].str.len().sum()
    assert len(cleaned) == len(results) - n_dropped

    pending = log.loc[log["resolving_step"] == STEP_NON_STARTER, ["raceId", "driverId"]]
    resolved = cleaned.merge(pending, on=["raceId", "driverId"], how="left", indicator=True)
    resolved = resolved[resolved["_merge"] == "left_only"]
    assert not resolved.duplicated(["raceId", "driverId"]).any()


def test_normal_rows_are_untouched(results, races):
    cleaned, _ = resolve_duplicate_results(results, races)
    pd.testing.assert_frame_equal(
        cleaned[cleaned["resultId"] == 60], results[results["resultId"] == 60]
    )


def test_input_is_not_modified(results, races):
    before = results.copy()
    resolve_duplicate_results(results, races)
    pd.testing.assert_frame_equal(results, before)


def test_lowest_grid_vs_lowest_resultid_flag(results, races):
    _, log = resolve_duplicate_results(results, races)
    flags = log.set_index("raceId")["lowest_grid_differs_from_lowest_resultId"]
    # race 2: kept 22 (lowest grid) but lowest resultId is 21 -> differs
    assert flags[2] == True  # noqa: E712  (pandas BooleanDtype scalar)
    assert flags[1] == False  # noqa: E712
    assert pd.isna(flags[4])


def test_duplicate_result_id_is_rejected(results, races):
    bad = pd.concat([results, results.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="resultId"):
        resolve_duplicate_results(bad, races)


# ---------------------------------------------------------------------------
# filter_non_starters (D3)
# ---------------------------------------------------------------------------

STATUS = pd.DataFrame(
    {
        "statusId": [1, 2, 3, 4, 5, 6],
        "status": ["Finished", "Did not qualify", "Did not prequalify", "107% Rule", "Withdrew", "Not classified"],
    }
).astype({"statusId": "Int64", "status": "string"})

# resultId, raceId, driverId, statusId, grid, laps.
# laps deliberately contradicts the expected outcome in rows 2 and 6, to prove
# the filter never uses it.
NON_START_ROWS = [
    (1, 1, 101, 2, 0, 0),    # a: Did not qualify, grid 0          -> excluded
    (2, 1, 102, 2, 12, 5),   # b: Did not qualify, grid > 0, laps  -> excluded
    (3, 1, 103, 3, 0, 0),    # c: Did not prequalify                -> excluded
    (4, 1, 104, 4, 20, 0),   # d: 107% Rule, grid > 0               -> excluded
    (5, 1, 105, 5, 0, 0),    # e: Withdrew, grid 0                  -> excluded
    (6, 1, 106, 5, 8, 0),    # f: Withdrew, grid > 0, no laps       -> kept
    (7, 1, 107, 6, 15, 40),  # g: Not classified                    -> kept
    (8, 1, 108, 1, 0, 57),   # h: Finished from the pit lane        -> kept
]


@pytest.fixture
def entries():
    cols = ["resultId", "raceId", "driverId", "statusId", "grid", "laps"]
    return pd.DataFrame(NON_START_ROWS, columns=cols).astype("Int64")


@pytest.fixture
def one_race():
    return pd.DataFrame({"raceId": [1], "year": [1985]}).astype("Int64")


def test_non_starter_filter_excludes_exactly_the_non_starts(entries, one_race):
    kept, log = filter_non_starters(entries, STATUS, one_race)
    assert sorted(log["resultId"]) == [1, 2, 3, 4, 5]
    assert sorted(kept["resultId"]) == [6, 7, 8]


def test_non_starter_reasons(entries, one_race):
    _, log = filter_non_starters(entries, STATUS, one_race)
    reasons = dict(zip(log["resultId"], log["reason"]))
    assert reasons == {
        1: "status: Did not qualify",
        2: "status: Did not qualify",
        3: "status: Did not prequalify",
        4: "status: 107% Rule",
        5: "Withdrew with grid = 0",
    }
    assert set(log["year"]) == {1985}


def test_withdrew_depends_on_grid_only(entries, one_race):
    kept, log = filter_non_starters(entries, STATUS, one_race)
    assert 5 in set(log["resultId"])   # Withdrew, grid 0
    assert 6 in set(kept["resultId"])  # Withdrew, grid > 0 (even with 0 laps)


def test_not_classified_and_pit_lane_start_are_kept(entries, one_race):
    kept, _ = filter_non_starters(entries, STATUS, one_race)
    assert {7, 8} <= set(kept["resultId"])


def test_non_starter_row_counts_and_input_unchanged(entries, one_race):
    before = entries.copy()
    kept, log = filter_non_starters(entries, STATUS, one_race)
    assert len(kept) + len(log) == len(entries)
    pd.testing.assert_frame_equal(entries, before)


def test_missing_status_name_is_rejected(entries, one_race):
    no_107 = STATUS[STATUS["status"] != "107% Rule"]
    with pytest.raises(ValueError, match="107% Rule"):
        filter_non_starters(entries, no_107, one_race)


# ---------------------------------------------------------------------------
# exclude_indy500 (D2)
# ---------------------------------------------------------------------------

def test_indy500_excluded_only_at_indianapolis_in_1950_1960():
    circuits = pd.DataFrame(
        {"circuitId": [19, 6], "circuitRef": ["indianapolis", "monaco"]}
    ).astype({"circuitId": "Int64", "circuitRef": "string"})
    races = pd.DataFrame({
        "raceId": [1, 2, 3, 4],
        "year": [1950, 1961, 2005, 1955],
        "circuitId": [19, 19, 19, 6],
        "name": ["Indianapolis 500", "Hypothetical 1961 race", "United States Grand Prix", "Monaco Grand Prix"],
    }).astype({"raceId": "Int64", "year": "Int64", "circuitId": "Int64", "name": "string"})
    results = pd.DataFrame({"resultId": [10, 11, 20, 30, 40], "raceId": [1, 1, 2, 3, 4]}).astype("Int64")

    kept, log = exclude_indy500(results, races, circuits)
    assert sorted(kept["resultId"]) == [20, 30, 40]
    assert log["raceId"].tolist() == [1]
    assert log["n_rows_excluded"].tolist() == [2]
