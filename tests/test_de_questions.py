"""Tests for the data-engineering helpers: Q1 (front-row conversion), Q2 (home advantage) and
Q3 (mechanical retirements per constructor)."""

import numpy as np
import pandas as pd
import pytest

from f1podium import config
from f1podium import de_questions as de
from f1podium.load import load_table


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


# ---------------------------------------------------------------------------
# Q2: home advantage
# ---------------------------------------------------------------------------

RAW_PRESENT = all((config.RAW_DIR / f"{t}.csv").exists() for t in ("drivers", "circuits"))


@pytest.mark.skipif(not RAW_PRESENT, reason="raw data not available")
def test_nationality_map_is_complete_for_the_real_data():
    mapping = de.load_nationality_map()
    check = de.check_nationality_map(mapping, load_table("drivers"), load_table("circuits"))
    assert check["unmapped_nationalities"] == []      # every driver nationality maps
    assert check["unknown_countries"] == []           # every country exists in circuits or is "no circuit"
    assert check["drivers_without_nationality"] == 0


def test_nationality_map_file_is_well_formed():
    mapping = de.load_nationality_map()
    assert not mapping.duplicated(["nationality", "country"]).any()
    assert set(mapping.loc[mapping["nationality"] == "American", "country"]) == {"USA", "United States"}
    assert set(mapping.loc[mapping["nationality"] == "Irish", "country"]) == {de.NO_CIRCUIT}


def test_check_nationality_map_flags_unmapped_and_unknown():
    mapping = pd.DataFrame({"nationality": ["British", "Martian"], "country": ["UK", "Mars"], "note": ["", ""]})
    drivers = pd.DataFrame({"nationality": ["British", "Irish "]})
    circuits = pd.DataFrame({"country": ["UK"]})
    out = de.check_nationality_map(mapping, drivers, circuits)
    assert out["unmapped_nationalities"] == ["Irish"]
    assert out["unknown_countries"] == ["Mars"]


@pytest.mark.parametrize("rows, message", [
    ("nationality,country,note\nBritish,UK,\nBritish,UK,\n", "duplicate"),
    ("nationality,country,note\nIrish,UK,\nIrish,no circuit,\n", "no circuit"),
    ("nationality,country\nBritish,UK\n", "columns"),
])
def test_load_nationality_map_rejects_bad_files(tmp_path, rows, message):
    path = tmp_path / "map.csv"
    path.write_text(rows, encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        de.load_nationality_map(path)


@pytest.fixture
def q2_tables():
    results = ints(
        [(1, 1, 1, 1), (2, 1, 5, 4), (3, 1, 2, 2), (3, 2, 3, 3), (3, 4, 0, 9), (1, 3, 7, 8)],
        ["raceId", "driverId", "grid", "positionOrder"],
    )
    races = ints([(1, 2010, 10), (2, 2023, 20), (3, 2015, 30)], ["raceId", "year", "circuitId"])
    circuits = pd.DataFrame({"circuitId": pd.array([10, 20, 30], dtype="Int64"),
                             "country": ["USA", "United States", "UK"]})
    drivers = pd.DataFrame({"driverId": pd.array([1, 2, 3, 4], dtype="Int64"),
                            "nationality": ["American", "Irish", "Argentinian ", "British"]})
    mapping = pd.DataFrame({
        "nationality": ["American", "American", "Irish", "Argentinian", "British"],
        "country": ["USA", "United States", de.NO_CIRCUIT, "Argentina", "UK"],
        "note": [""] * 5,
    })
    return results, races, circuits, drivers, mapping


def test_home_table_keeps_rows_and_flags_home(q2_tables):
    results = q2_tables[0]
    h = de.build_home_table(*q2_tables)
    assert len(h) == len(results)                      # joins (incl. a 2-country nationality) add no rows
    home = h.set_index(["raceId", "driverId"])["home"]
    assert home[(1, 1)] and home[(2, 1)]                # American at 'USA' and at 'United States'
    assert not home[(3, 2)]                             # Irish at a UK circuit: not home
    assert home[(3, 4)]                                 # British at a UK circuit
    assert not home[(1, 3)]                             # Argentinian at a US circuit
    assert (h["nationality"] == "Argentinian").sum() == 1  # trailing space stripped before mapping
    assert h.set_index(["raceId", "driverId"]).loc[(3, 4), "grid_bucket"] == config.GRID_BUCKET_PIT_LANE


@pytest.mark.parametrize("which", ["races", "circuits", "drivers", "mapping"])
def test_home_table_rejects_duplicate_keys(q2_tables, which):
    t = dict(zip(["results", "races", "circuits", "drivers", "mapping"], q2_tables))
    t[which] = pd.concat([t[which], t[which].iloc[[0]]], ignore_index=True)
    with pytest.raises(AssertionError, match="not unique"):
        de.build_home_table(t["results"], t["races"], t["circuits"], t["drivers"], t["mapping"])


@pytest.mark.parametrize("grid, bucket", [
    (0, config.GRID_BUCKET_PIT_LANE), (1, "1"), (2, "2–3"), (3, "2–3"), (4, "4–6"),
    (6, "4–6"), (7, "7–10"), (10, "7–10"), (11, "11+"), (26, "11+"),
])
def test_grid_bucket_edges(grid, bucket):
    assert de.assign_grid_bucket(pd.Series(pd.array([grid], dtype="Int64")))[0] == bucket


def test_grid_bucket_missing_and_invalid():
    out = de.assign_grid_bucket(pd.Series(pd.array([None, 1], dtype="Int64")))
    assert pd.isna(out[0]) and out[1] == "1"
    with pytest.raises(ValueError, match="non-negative"):
        de.assign_grid_bucket(pd.Series(pd.array([-1], dtype="Int64")))


def two_by_two(a, b, c, d, bucket):
    """Rows for one stratum: a home podiums, b home non-podiums, c away podiums, d away non-podiums."""
    return pd.DataFrame({
        "home": [True] * (a + b) + [False] * (c + d),
        "podium": [True] * a + [False] * b + [True] * c + [False] * d,
        "grid_bucket": bucket,
    })


def test_cmh_odds_ratio_single_stratum():
    out = de.cmh_home_or(two_by_two(10, 10, 5, 15, "1"))
    assert out["odds_ratio"] == pytest.approx(3.0)       # (10 * 15) / (10 * 5)
    assert out["ci_low"] < 3.0 < out["ci_high"]
    assert np.isnan(out["homogeneity_p"])                # needs two or more strata


def test_cmh_identical_strata_and_uninformative_stratum():
    df = pd.concat([
        two_by_two(10, 10, 5, 15, "1"),
        two_by_two(10, 10, 5, 15, "2–3"),
        two_by_two(0, 4, 0, 30, config.GRID_BUCKET_PIT_LANE),  # no podiums: no information on the OR
    ], ignore_index=True)
    out = de.cmh_home_or(df)
    assert out["odds_ratio"] == pytest.approx(3.0)
    assert out["homogeneity_p"] == pytest.approx(1.0)
    assert out["strata_dropped"] == [config.GRID_BUCKET_PIT_LANE]
    assert out["n_rows"] == 80


# ---------------------------------------------------------------------------
# Q3: mechanical retirements per constructor
# ---------------------------------------------------------------------------

Q3_RAW_PRESENT = all((config.RAW_DIR / f"{t}.csv").exists() for t in ("status", "results", "races"))


@pytest.mark.skipif(not Q3_RAW_PRESENT, reason="raw data not available")
def test_status_mapping_covers_every_status_in_the_real_data():
    mapping = de.load_status_categories()
    status = load_table("status")
    assert de.check_status_categories(mapping, status) == {"unmapped_statuses": [], "unknown_statuses": []}
    assert len(mapping) == len(status)


def test_status_mapping_file_is_well_formed():
    mapping = de.load_status_categories()
    assert mapping["status"].is_unique
    assert set(mapping["category"]) <= set(de.STATUS_CATEGORIES)
    amb = mapping[mapping["ambiguous"]]
    assert (amb["alt_category"] != amb["category"]).all()
    assert (mapping.loc[~mapping["ambiguous"], "alt_category"] == mapping.loc[~mapping["ambiguous"], "category"]).all()
    cat = mapping.set_index("status")["category"]
    assert cat["Finished"] == "finished" and cat["+1 Lap"] == "finished"
    assert cat["Engine"] == de.MECHANICAL and cat["Collision"] == "accident_driver"
    assert cat["Disqualified"] == "regulatory" and cat["Retired"] == "other"


@pytest.mark.parametrize("rows, message", [
    ("status,category,ambiguous,alt_category,note\nEngine,mechanical,False,,\nEngine,mechanical,False,,\n", "duplicate"),
    ("status,category,ambiguous,alt_category,note\nEngine,broken,False,,\n", "unknown category"),
    ("status,category,ambiguous,alt_category,note\nRetired,other,True,,\n", "alt_category"),
    ("status,category,ambiguous,alt_category,note\nEngine,mechanical,False,other,\n", "alt_category"),
    ("status,category,ambiguous,alt_category,note\nRetired,other,True,other,\n", "alt_category"),
    ("status,category\nEngine,mechanical\n", "columns"),
])
def test_load_status_categories_rejects_bad_files(tmp_path, rows, message):
    path = tmp_path / "map.csv"
    path.write_text(rows, encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        de.load_status_categories(path)


def lineage_of(ref_years, constructors):
    """Lineage for a list of (constructorRef, year) pairs, via de.lineage_for."""
    ids = constructors.set_index("constructorRef")["constructorId"]
    df = pd.DataFrame({"constructorId": [ids[r] for r, _ in ref_years], "year": [y for _, y in ref_years]})
    return de.lineage_for(df, de.load_constructor_lineage(), constructors).tolist()


@pytest.mark.skipif(not Q3_RAW_PRESENT, reason="raw data not available")
def test_lineage_merges_renamed_teams_and_separates_reused_ids():
    cons = load_table("constructors")
    rb = lineage_of([("minardi", 2005), ("toro_rosso", 2019), ("alphatauri", 2020), ("rb", 2024)], cons)
    assert len(set(rb)) == 1 and rb[0] == "Minardi / Toro Rosso / AlphaTauri / RB"
    enstone = lineage_of([("renault", 2020), ("alpine", 2021), ("renault", 2005), ("lotus_f1", 2014)], cons)
    assert len(set(enstone)) == 1
    assert lineage_of([("renault", 1980)], cons) == ["Renault"]            # 1977-85 works team kept apart
    am = lineage_of([("force_india", 2018), ("racing_point", 2019), ("aston_martin", 2021)], cons)
    assert len(set(am)) == 1
    assert lineage_of([("aston_martin", 1959)], cons) == ["Aston Martin"]  # 1959-60 team kept apart
    assert lineage_of([("mercedes", 1954), ("mercedes", 2014)], cons)[0] == "Mercedes"
    assert lineage_of([("mclaren", 2024), ("ferrari", 1950)], cons) == ["McLaren", "Ferrari"]  # not covered


@pytest.mark.skipif(not Q3_RAW_PRESENT, reason="raw data not available")
def test_every_result_resolves_to_one_lineage():
    cons = load_table("constructors")
    results = load_table("results").merge(load_table("races")[["raceId", "year"]], on="raceId")
    lin = de.lineage_for(results, de.load_constructor_lineage(), cons)
    assert lin.notna().all() and len(lin) == len(results)
    # Explicit lineage names never collide with a constructor name used as a fallback.
    explicit = set(de.load_constructor_lineage()["lineage"])
    assert not explicit & set(cons["name"])


@pytest.mark.parametrize("rows, message", [
    ("constructorId,constructorRef,first_year,last_year,lineage,note\n4,renault,2002,2011,E,\n4,renault,2010,2020,E,\n",
     "overlapping"),
    ("constructorId,constructorRef,first_year,last_year,lineage,note\n4,renault,2012,2011,E,\n", "first_year"),
    ("constructorId,constructorRef,lineage,note\n4,renault,E,\n", "columns"),
])
def test_load_constructor_lineage_rejects_bad_files(tmp_path, rows, message):
    path = tmp_path / "lineage.csv"
    path.write_text(rows, encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        de.load_constructor_lineage(path)


@pytest.fixture
def q3_tables():
    # raceId 1: 2016; raceId 2: 2023; raceId 3: 2010 (outside both eras).
    results = ints(
        [
            # raceId, driverId, constructorId, statusId
            (1, 1, 100, 1),   # finished
            (1, 2, 100, 5),   # engine -> mechanical
            (1, 3, 200, 31),  # retired -> other (ambiguous, alt mechanical)
            (1, 4, 200, 4),   # collision
            (2, 1, 100, 5),
            (2, 2, 200, 1),
            (3, 1, 100, 5),   # outside the eras: dropped after the joins
        ],
        ["raceId", "driverId", "constructorId", "statusId"],
    )
    races = ints([(1, 2016), (2, 2023), (3, 2010)], ["raceId", "year"])
    constructors = pd.DataFrame({"constructorId": pd.array([100, 200], dtype="Int64"), "name": ["Alpha", "Beta"]})
    status = pd.DataFrame({"statusId": pd.array([1, 4, 5, 31], dtype="Int64"),
                           "status": ["Finished", "Collision", "Engine", "Retired"]})
    return results, races, constructors, status, de.load_status_categories()


def test_retirement_table_keeps_rows_and_flags_mechanical(q3_tables):
    out = de.build_retirement_table(*q3_tables)
    assert len(out) == 6                                         # 7 rows joined, 1 outside the eras
    assert not out.duplicated(de.KEY).any()
    assert out.groupby("era").size().to_dict() == {"2014–2021": 4, "2022–2024": 2}
    assert out["mechanical"].sum() == 2
    assert out["mechanical_alt"].sum() == 3                      # Retired flips to mechanical
    assert out.loc[out["status"] == "Retired", "ambiguous"].all()


@pytest.mark.parametrize("break_it, message", [
    (lambda r, s: (pd.concat([r, r.iloc[[0]]]), s), "not unique"),
    (lambda r, s: (r.assign(statusId=r["statusId"].mask(r.index == 0)), s), "null statusId"),
    (lambda r, s: (r, s.assign(status=s["status"].replace("Engine", "Warp core"))), "not in the mapping"),
    (lambda r, s: (r, s[s["statusId"] != 5]), "statusId not in status"),
])
def test_retirement_table_rejects_bad_inputs(q3_tables, break_it, message):
    results, races, constructors, status, mapping = q3_tables
    results, status = break_it(results, status)
    with pytest.raises(AssertionError, match=message):
        de.build_retirement_table(results, races, constructors, status, mapping)


def test_retirement_table_join_row_count_check(q3_tables):
    results, races, constructors, status, mapping = q3_tables
    dup_status = pd.concat([status, status.iloc[[0]]])          # a duplicated statusId would multiply rows
    with pytest.raises(AssertionError, match="not unique"):
        de.build_retirement_table(results, races, constructors, dup_status, mapping)


def test_mechanical_rates_counts_and_minimum(q3_tables):
    out = de.build_retirement_table(*q3_tables)
    rates = de.mechanical_rates(out, min_starts=2).set_index(["constructorId", "era"])
    alpha_old = rates.loc[(100, "2014–2021")]
    assert (alpha_old["starts"], alpha_old["n_mech"], alpha_old["rate"]) == (2, 1, 0.5)
    assert alpha_old["label"] == "Alpha" and alpha_old["meets_min"]
    assert not rates.loc[(100, "2022–2024"), "meets_min"]       # 1 start < 2
    assert rates.loc[(200, "2014–2021"), "n_mech"] == 0


def test_wilson_interval_known_case_for_q3_rates():
    low, high = de.wilson_interval(3, 40)                       # 7.5%
    assert low == pytest.approx(0.0258, abs=1e-4)
    assert high == pytest.approx(0.1986, abs=1e-4)


def test_overall_mechanical_change_counts(q3_tables):
    out = de.build_retirement_table(*q3_tables)
    res = de.overall_mechanical_change(out)
    assert (res["starts_old"], res["mech_old"], res["starts_new"], res["mech_new"]) == (4, 1, 2, 1)
    assert res["diff"] == pytest.approx(0.5 - 0.25)
    alt = de.overall_mechanical_change(out, flag="mechanical_alt")
    assert alt["mech_old"] == 2


def toy_lineage():
    return pd.DataFrame({"constructorId": pd.array([100], dtype="Int64"), "first_year": pd.array([2015], dtype="Int64"),
                         "last_year": pd.array([2024], dtype="Int64"), "lineage": ["Alpha line"]})


def test_add_lineage_uses_year_range_and_falls_back_to_name(q3_tables):
    out = de.build_retirement_table(*q3_tables)
    constructors = q3_tables[2]
    lin = de.add_lineage(out, toy_lineage(), constructors).set_index(["raceId", "driverId"])["lineage"]
    assert lin[(1, 1)] == "Alpha line" and lin[(2, 1)] == "Alpha line"   # constructor 100 in 2016 and 2023
    assert lin[(1, 3)] == "Beta"                                         # constructor 200: not covered -> own name
    assert len(lin) == len(out)


def test_add_lineage_raises_on_unknown_constructor(q3_tables):
    out = de.build_retirement_table(*q3_tables)
    constructors = q3_tables[2][q3_tables[2]["constructorId"] != 200]
    with pytest.raises(AssertionError, match="not in constructors"):
        de.add_lineage(out, toy_lineage(), constructors)
