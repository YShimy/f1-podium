"""Tests for resolve_duplicate_results on a small fake results table."""

import pandas as pd
import pytest

from f1podium.clean import (
    STEP_LOWEST_GRID,
    STEP_LOWEST_RESULT_ID,
    STEP_NON_STARTER,
    STEP_STARTED_ROW,
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
