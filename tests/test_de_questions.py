"""Tests for the data-engineering Q1 helpers (front-row conversion by circuit)."""

import numpy as np
import pandas as pd
import pytest

from f1podium import de_questions as de


def ints(rows, cols):
    return pd.DataFrame(rows, columns=cols).astype("Int64")


@pytest.fixture
def tables():
    # raceId 1: 2005 at circuit 10; raceId 2: 2016 at circuit 20.
    results = ints(
        [
            # raceId, driverId, grid, positionOrder
            (1, 1, 1, 1),   # pole, wins
            (1, 2, 2, 5),   # front row, no podium
            (1, 3, 3, 2),
            (1, 4, 0, 9),   # pit-lane start: never front row
            (2, 1, 5, 3),   # qualified P1 but started P5 (penalty)
            (2, 2, 1, 4),   # started P1 after others' penalties; qualified P4
            (2, 5, 2, 1),   # no qualifying row at all
        ],
        ["raceId", "driverId", "grid", "positionOrder"],
    )
    races = ints([(1, 2005, 1, 10), (2, 2016, 3, 20)], ["raceId", "year", "round", "circuitId"])
    circuits = pd.DataFrame({"circuitId": pd.array([10, 20], dtype="Int64"), "name": ["Alpha", "Beta"]})
    qualifying = ints(
        [(1, 1, 1), (1, 2, 2), (1, 3, 3), (1, 4, 4), (2, 1, 1), (2, 2, 4), (9, 9, 1)],  # last row: no result
        ["raceId", "driverId", "position"],
    )
    return results, races, circuits, qualifying


def test_wilson_interval_known_values():
    low, high = de.wilson_interval(5, 10)
    assert low == pytest.approx(0.2366, abs=1e-3)
    assert high == pytest.approx(0.7634, abs=1e-3)

    low, high = de.wilson_interval(0, 10)
    assert low == pytest.approx(0.0, abs=1e-9)
    assert high == pytest.approx(0.2775, abs=1e-3)


def test_wilson_interval_vectorised_and_empty():
    low, high = de.wilson_interval([5, 0], [10, 0])
    assert low[0] == pytest.approx(0.2366, abs=1e-3)
    assert np.isnan(low[1]) and np.isnan(high[1])
    assert (low <= high).all() or np.isnan(low).any()


def test_front_row_grid_is_never_more_than_two_per_race_and_never_grid_zero(tables):
    fr = de.build_front_row_table(*tables)
    assert de.races_over_front_row_limit(fr, "front_row_grid").empty
    assert fr.loc[fr["grid"] == 0, "front_row_grid"].eq(False).all()
    assert fr[fr["front_row_grid"]].groupby("raceId").size().max() <= 2


def test_front_row_limit_check_catches_a_violation(tables):
    results, races, circuits, qualifying = tables
    bad = pd.concat([results, ints([(1, 6, 1, 7)], list(results.columns))], ignore_index=True)  # 3rd grid-1/2 car
    fr = de.build_front_row_table(bad, races, circuits, qualifying)
    violations = de.races_over_front_row_limit(fr, "front_row_grid")
    assert violations["raceId"].tolist() == [1]
    assert violations["n_front_row"].tolist() == [3]


def test_joins_do_not_multiply_rows(tables):
    results, races, circuits, qualifying = tables
    fr = de.build_front_row_table(results, races, circuits, qualifying)
    assert len(fr) == len(results)  # the extra qualifying row (raceId 9) is ignored by the left join
    assert not fr.duplicated(["raceId", "driverId"]).any()
    assert fr.loc[fr["raceId"] == 1, "circuit"].eq("Alpha").all()


@pytest.mark.parametrize("which", ["qualifying", "circuits", "races", "results"])
def test_duplicate_keys_are_rejected(tables, which):
    results, races, circuits, qualifying = tables
    t = {"results": results, "races": races, "circuits": circuits, "qualifying": qualifying}
    t[which] = pd.concat([t[which], t[which].iloc[[0]]], ignore_index=True)
    with pytest.raises(AssertionError, match="not unique"):
        de.build_front_row_table(t["results"], t["races"], t["circuits"], t["qualifying"])


def test_quali_and_grid_front_row_are_distinct(tables):
    fr = de.build_front_row_table(*tables).set_index(["raceId", "driverId"])
    penalised = fr.loc[(2, 1)]    # quali P1, grid P5
    promoted = fr.loc[(2, 2)]     # quali P4, grid P1
    no_quali = fr.loc[(2, 5)]     # grid P2, no qualifying row
    assert penalised["front_row_quali"] and not penalised["front_row_grid"]
    assert promoted["front_row_grid"] and not promoted["front_row_quali"]
    assert no_quali["front_row_grid"] and not no_quali["front_row_quali"]
    assert pd.isna(no_quali["quali_position"])


def test_circuit_conversion_counts_eras_and_minimum():
    df = pd.DataFrame({
        "circuitId": [1] * 4 + [2] * 2,
        "circuit": ["A"] * 4 + ["B"] * 2,
        "year": [2013, 2013, 2014, 2014, 2014, 2014],
        "front_row_grid": [True] * 6,
        "podium": [True, False, True, True, False, False],
    })
    out = de.circuit_conversion(df, "front_row_grid", min_starts=2).set_index(["circuit", "era"])
    assert out.loc[("A", de.ERA_OLD), ["n_front_row", "n_podium"]].tolist() == [2, 1]
    assert out.loc[("A", de.ERA_NEW), ["n_front_row", "n_podium"]].tolist() == [2, 2]  # 2014 is the new era
    assert out.loc[("B", de.ERA_NEW), "rate"] == 0.0
    assert out["meets_min"].all()
    strict = de.circuit_conversion(df, "front_row_grid", min_starts=3)
    assert not strict["meets_min"].any()


def test_threshold_report_counts_dropped_circuits():
    conv = pd.DataFrame({
        "window": "w", "definition": "grid", "era": de.ERA_OLD,
        "circuitId": [1, 2, 3], "circuit": list("ABC"), "n_front_row": [4, 10, 12],
    })
    rep = de.threshold_report(conv, (5, 10)).set_index("min_starts")
    assert rep.loc[5, "circuits_dropped"] == 1 and rep.loc[5, "starts_in_dropped"] == 4
    assert rep.loc[10, "circuits_dropped"] == 1 and rep.loc[10, "circuits_kept"] == 2


def test_era_rank_correlation_uses_only_circuits_in_both_eras():
    rows = []
    for cid, (old, new) in {1: (0.1, 0.2), 2: (0.5, 0.6), 3: (0.9, 0.95)}.items():
        rows += [(cid, de.ERA_OLD, old, 20), (cid, de.ERA_NEW, new, 20)]
    rows.append((4, de.ERA_OLD, 0.3, 20))  # only in the old era
    conv = pd.DataFrame(rows, columns=["circuitId", "era", "rate", "n_front_row"]).assign(window="w", definition="grid")
    out = de.era_rank_correlation(conv).iloc[0]
    assert out["n_circuits_both_eras"] == 3
    assert out["spearman_both_eras"] == pytest.approx(1.0)


def test_quali_coverage_by_season(tables):
    fr = de.build_front_row_table(*tables)
    cov = de.quali_coverage_by_season(fr, window=(2003, 2024))
    assert cov.loc[2005, "coverage"] == 1.0
    assert cov.loc[2016, "n_rows"] == 3 and cov.loc[2016, "n_with_quali"] == 2


def test_spearman_interval_known_value_and_small_n():
    low, high = de.spearman_interval(0.5, 14)
    assert low == pytest.approx(-0.077, abs=2e-3)
    assert high == pytest.approx(0.826, abs=2e-3)
    assert all(np.isnan(v) for v in de.spearman_interval(0.9, 3))
    assert de.spearman_interval(1.0, 10)[1] == 1.0  # a perfect correlation does not raise


def test_ci_separated_pairs():
    n_pairs, n_separated = de.ci_separated_pairs([0.0, 0.5, 0.6], [0.2, 0.7, 0.9])
    assert (n_pairs, n_separated) == (3, 2)  # A is clear of B and C; B and C overlap


def test_two_proportion_test_known_case():
    out = de.two_proportion_test(50, 100, 60, 100)  # 50% -> 60%
    assert out["diff"] == pytest.approx(0.10)
    assert out["z"] == pytest.approx(1.4213, abs=1e-3)
    assert out["p_value"] == pytest.approx(0.1552, abs=1e-3)
    assert out["ci_low"] == pytest.approx(-0.037, abs=5e-3)   # Newcombe interval
    assert out["ci_high"] == pytest.approx(0.232, abs=5e-3)
    assert out["ci_low"] < 0 < out["ci_high"]                  # interval and p-value agree: not significant


def test_two_proportion_test_edge_cases():
    same = de.two_proportion_test(10, 20, 10, 20)
    assert same["diff"] == 0 and same["p_value"] == pytest.approx(1.0)
    none = de.two_proportion_test(0, 20, 0, 30)  # pooled variance is zero
    assert none["p_value"] == 1.0 and none["diff"] == 0
    big = de.two_proportion_test(20, 1000, 200, 1000)
    assert big["p_value"] < 1e-10 and big["ci_low"] > 0


def test_overall_era_difference_counts_by_definition():
    df = pd.DataFrame({
        "year": [2005, 2005, 2010, 2010, 2016, 2016, 2020, 2020],
        "front_row_grid": [True] * 8,
        "front_row_quali": [True, True, True, False, True, True, True, True],
        "podium": [True, False, True, True, True, True, False, True],
    })
    out = de.overall_era_difference(df).set_index("definition")
    assert out.loc["grid", ["starts_old", "podiums_old", "starts_new", "podiums_new"]].tolist() == [4, 3, 4, 3]
    assert out.loc["quali", ["starts_old", "starts_new"]].tolist() == [3, 4]
