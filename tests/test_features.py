"""Tests for the modeling table (features.py) and the six leakage checks (leakage.py).

Synthetic tests cover the lag logic with deliberately non-chronological raceIds. The real-data tests
(skipped without data/raw) run the full leakage checks on the actual table.
"""

import numpy as np
import pandas as pd
import pytest

from f1podium import config
from f1podium import features as F
from f1podium import leakage as L

RAW_PRESENT = all((config.RAW_DIR / f"{t}.csv").exists()
                  for t in ("results", "races", "qualifying", "sprint_results", "driver_standings"))


# ---------------------------------------------------------------------------
# Synthetic: lag logic
# ---------------------------------------------------------------------------

def toy_sequence():
    # raceIds deliberately NOT in chronological order: 30 is the first race, 10 the last.
    return pd.DataFrame({
        "driverId": [1, 1, 1, 1],
        "raceId":   [10, 30, 20, 40],
        "year":     [2001, 2000, 2000, 2001],
        "round":    [2, 1, 2, 1],
        "x":        [4.0, 1.0, 2.0, 3.0],  # chronological order: 1, 2, 3, 4
    })


def test_rolling_prior_orders_by_year_round_and_excludes_current():
    out = F.rolling_prior(toy_sequence(), "driverId", ["x"], (2,)).set_index("raceId")
    assert out.loc[30, "races_before"] == 0 and np.isnan(out.loc[30, "x_sum2"])   # first race: nothing before
    assert out.loc[20, "x_sum2"] == 1.0                                            # only race 30
    assert out.loc[40, "x_sum2"] == 1.0 + 2.0
    assert out.loc[10, "x_sum2"] == 2.0 + 3.0                                      # window of 2, current excluded
    assert out.loc[10, "races_before"] == 3


def test_season_points_before_resets_each_season_and_excludes_current_round():
    events = pd.DataFrame({
        "driverId": [1, 1, 1, 1, 1],
        "year":     [2000, 2000, 2000, 2001, 2001],
        "round":    [2, 1, 2, 1, 2],
        "pts":      [6.0, 10.0, 3.0, 8.0, 1.0],   # round 2 of 2000 has race (6) + sprint (3) points
    })
    out = F.season_points_before(events, "driverId").set_index(["year", "round"])["points_before"]
    assert out[(2000, 1)] == 0 and out[(2000, 2)] == 10
    assert out[(2001, 1)] == 0 and out[(2001, 2)] == 8


def test_assign_split_uses_config_years():
    years = pd.Series([1950, config.TRAIN_END_YEAR, config.VAL_YEARS[0], config.VAL_YEARS[1],
                       config.TEST_START_YEAR, 2024])
    assert F.assign_split(years).tolist() == ["train", "train", "validation", "validation", "test", "test"]


def synthetic_tables():
    """One season, raceIds deliberately non-chronological (30 = round 1, 10 = round 2, 20 = round 3).

    Constructor 7 ("Gamma") runs drivers 1 and 2; constructor 8 ("Delta") runs drivers 3 and 4.
    """
    rows = [  # raceId, driverId, constructorId, grid, positionOrder, points, statusId
        (30, 1, 7, 1, 1, 10, 1), (30, 2, 7, 2, 2, 6, 1), (30, 3, 8, 3, 3, 4, 1), (30, 4, 8, 4, 4, 3, 2),
        (10, 1, 7, 1, 3, 4, 1), (10, 2, 7, 2, 1, 10, 1), (10, 3, 8, 3, 2, 6, 1), (10, 4, 8, 4, 4, 3, 1),
        (20, 1, 7, 1, 1, 10, 1), (20, 2, 7, 2, 2, 6, 1), (20, 3, 8, 3, 3, 4, 1), (20, 4, 8, 4, 4, 3, 1),
    ]
    rc = pd.DataFrame(rows, columns=["raceId", "driverId", "constructorId", "grid", "positionOrder", "points",
                                     "statusId"]).astype("Int64")
    tables = {
        "races": pd.DataFrame({"raceId": [30, 10, 20], "year": [2000] * 3, "round": [1, 2, 3],
                               "circuitId": [1] * 3}).astype("Int64"),
        "qualifying": rc[["raceId", "driverId", "grid"]].rename(columns={"grid": "position"}),
        "sprint_results": pd.DataFrame({c: pd.array([], dtype="Int64")
                                        for c in ["raceId", "driverId", "constructorId", "points"]}),
        "circuits": pd.DataFrame({"circuitId": pd.array([1], dtype="Int64"), "country": ["UK"]}),
        "drivers": pd.DataFrame({"driverId": pd.array([1, 2, 3, 4], dtype="Int64"), "nationality": ["British"] * 4}),
        "constructors": pd.DataFrame({"constructorId": pd.array([7, 8], dtype="Int64"), "name": ["Gamma", "Delta"]}),
        "status": pd.DataFrame({"statusId": pd.array([1, 2], dtype="Int64"), "status": ["Finished", "Engine"]}),
    }
    nat_map = pd.DataFrame({"nationality": ["British"], "country": ["UK"], "note": [""]})
    return tables, rc, nat_map


def synthetic_build(tables, rc, nat_map, key):
    meta, X, _ = F.build_modeling_table(tables, rc, nat_map=nat_map, constructor_key=key)
    return pd.concat([meta, X], axis=1).set_index(["round", "driverId"])


@pytest.mark.parametrize("key", F.CONSTRUCTOR_KEYS)
def test_teammate_results_in_the_same_race_never_enter_constructor_features(key):
    tables, rc, nat_map = synthetic_tables()
    d = synthetic_build(tables, rc, nat_map, key)
    con_form = [c for c in F.FEATURES if c.startswith("con_") and c not in ("con_points_before", "con_rank_before",
                                                                            "con_races_before")]
    # Round 1: both Gamma cars finished on the podium in this race, yet no constructor feature sees it.
    for drv in (1, 2):
        assert d.loc[(1, drv), con_form].isna().all()
        assert d.loc[(1, drv), "con_points_before"] == 0 and d.loc[(1, drv), "con_races_before"] == 0
    # Round 2: only round 1 counts (2 podiums from 2 starts; positions 1 and 2; 16 points).
    for drv in (1, 2):
        assert d.loc[(2, drv), "con_podium_rate_last3"] == 1.0
        assert d.loc[(2, drv), "con_mean_pos_last3"] == 1.5
        assert d.loc[(2, drv), "con_points_before"] == 16
        assert d.loc[(2, drv), "con_races_before"] == 1
        assert d.loc[(2, drv), F.MECH_FEATURE] == 0.0
    assert d.loc[(2, 3), F.MECH_FEATURE] == 0.5          # Delta: driver 4's engine failure in round 1

    # Change round 2: Gamma's winner (driver 2) and Delta's driver 4 swap results.
    pert = rc.copy()
    a = (pert["raceId"] == 10) & (pert["driverId"] == 2)
    b = (pert["raceId"] == 10) & (pert["driverId"] == 4)
    for col in ("positionOrder", "points", "statusId"):
        va, vb = pert.loc[a, col].iloc[0], pert.loc[b, col].iloc[0]
        pert.loc[a, col], pert.loc[b, col] = vb, va
    d2 = synthetic_build(tables, pert, nat_map, key)
    feats = F.FEATURES
    assert d.loc[[1, 2], feats].equals(d2.loc[[1, 2], feats])             # rounds 1-2 unchanged
    assert d.loc[(3, 1), "con_podium_rate_last3"] != d2.loc[(3, 1), "con_podium_rate_last3"]  # round 3 sees it


def test_feature_dictionary_matches_features_and_avoids_forbidden_names():
    fd = F.feature_dictionary()
    assert fd.index.tolist() == F.FEATURES and fd.index.is_unique
    assert not set(F.FEATURES) & set(config.FORBIDDEN_CURRENT_RACE_COLUMNS)
    assert set(L.LAGGED_FEATURES) <= set(F.FEATURES)


# ---------------------------------------------------------------------------
# Real data: the six leakage checks
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def real():
    from f1podium.clean import clean_results
    from f1podium.load import load_all
    tables = load_all()
    rc, _ = clean_results(tables["results"], tables["races"], tables["circuits"], tables["status"])
    meta, X, y = F.build_modeling_table(tables, rc)
    return tables, rc, meta, X, y


def race_id(meta, year, rnd):
    return int(meta.loc[(meta["year"] == year) & (meta["round"] == rnd), "raceId"].iloc[0])


@pytest.mark.skipif(not RAW_PRESENT, reason="raw data not available")
def test_grain_and_shapes(real):
    _, rc, meta, X, y = real
    assert len(meta) == len(X) == len(y) == len(rc)
    assert not meta.duplicated(["raceId", "driverId"]).any()
    assert meta.index.equals(X.index) and X.index.equals(y.index)
    assert list(X.columns) == F.FEATURES
    assert set(y.unique()) <= {0, 1}


@pytest.mark.skipif(not RAW_PRESENT, reason="raw data not available")
def test_1_bruteforce_recomputation_of_200_random_rows(real):
    tables, rc, meta, X, _ = real
    mism, n = L.bruteforce_mismatches(tables, rc, meta, X, n=200)
    assert n == 200
    assert mism.empty, mism.head(20)


@pytest.mark.skipif(not RAW_PRESENT, reason="raw data not available")
def test_1_bruteforce_constructor_id_ablation(real):
    tables, rc, *_ = real
    meta, X, _ = F.build_modeling_table(tables, rc, constructor_key="constructorId")
    mism, n = L.bruteforce_mismatches(tables, rc, meta, X, n=40, constructor_key="constructorId")
    assert n == 40 and mism.empty, mism.head(20)


def first_race_row(meta, X, tables, year, name):
    cid = int(tables["constructors"].set_index("name")["constructorId"][name])
    d = pd.concat([meta, X], axis=1)
    return d[(d["year"] == year) & (d["constructorId"] == cid)].sort_values("round").iloc[0]


@pytest.mark.skipif(not RAW_PRESENT, reason="raw data not available")
@pytest.mark.parametrize("year, name, members", [
    (2024, "RB F1 Team", {"minardi", "toro_rosso", "alphatauri"}),
    (2021, "Alpine F1 Team", {"toleman", "benetton", "renault", "lotus_f1"}),
    (2021, "Aston Martin", {"jordan", "mf1", "spyker_mf1", "spyker", "force_india", "racing_point"}),
])
def test_lineage_history_continues_across_renames(real, year, name, members):
    tables, rc, meta, X, _ = real
    row = first_race_row(meta, X, tables, year, name)
    refs = tables["constructors"].set_index("constructorId")["constructorRef"]
    r = rc.merge(tables["races"][["raceId", "year", "round"]], on="raceId")
    earlier = r[(r["year"] < year) & r["constructorId"].map(refs).isin(members)]
    if "renault" in members:                                  # the 1977-85 Renault works team is not Enstone
        earlier = earlier[~((earlier["constructorId"].map(refs) == "renault") & (earlier["year"] <= 1985))]
    assert row["con_races_before"] == earlier["raceId"].nunique() > 0

    # Ablation (constructorId): history is whatever that id had before -- 0 for RB and Alpine; for Aston
    # Martin the 5 races of the unrelated 1959-60 team, because Ergast reuses id 117.
    meta_id, X_id, _ = F.build_modeling_table(tables, rc, constructor_key="constructorId")
    row_id = first_race_row(meta_id, X_id, tables, year, name)
    same_id = r[(r["year"] < year) & (r["constructorId"] == row_id["constructorId"])]
    assert row_id["con_races_before"] == same_id["raceId"].nunique() < row["con_races_before"]


@pytest.mark.skipif(not RAW_PRESENT, reason="raw data not available")
def test_1b_bruteforce_on_edge_rows(real):
    # Round 1 rows, career debuts and a sprint weekend's next round: the cases most likely to go wrong.
    tables, rc, meta, X, _ = real
    edge = meta.index[(meta["round"] == 1) & (meta["year"].isin([1950, 2022]))].tolist()[:10]
    edge += X.index[X["drv_races_before"] == 0].tolist()[-5:]
    edge += meta.index[meta["raceId"] == race_id(meta, 2021, 11)].tolist()[:5]   # after the first sprint (R10)
    mism, _ = L.bruteforce_mismatches(tables, rc, meta, X, rows_idx=edge)
    assert mism.empty, mism


@pytest.mark.skipif(not RAW_PRESENT, reason="raw data not available")
@pytest.mark.parametrize("year, rnd", [(2012, 10), (1985, 1)])
def test_2_perturbing_race_r_changes_only_later_races(real, year, rnd):
    tables, rc, meta, _, _ = real
    out = L.perturbation_check(tables, rc, race_id(meta, year, rnd))
    assert out["changed_rows_up_to_and_incl_R"] == 0, out
    assert out["changed_rows_after_R"] > 0, out
    assert out["changed_targets_in_R"] > 0 and out["changed_targets_outside_R"] == 0, out


@pytest.mark.skipif(not RAW_PRESENT, reason="raw data not available")
def test_3_points_before_equals_previous_round_standings(real):
    tables, _, meta, X, _ = real
    check = L.standings_check(tables, meta, X)
    drivers = check[(check["who"] == "drivers") & (check["decade"] >= 1990)]
    assert drivers["mismatches"].sum() == 0, drivers          # sprints of earlier rounds included
    # Constructors from 1980: only documented standings adjustments may differ.
    races = tables["races"][["raceId", "year", "round"]]
    prev = tables["constructor_standings"].merge(races, on="raceId")[["constructorId", "year", "round", "points"]]
    prev = prev.assign(round=prev["round"] + 1)
    rows = pd.concat([meta, X[["con_points_before"]]], axis=1)
    rows = rows[(rows["round"] > 1) & (rows["year"] >= 1980)].drop_duplicates(["raceId", "constructorId"])
    m = rows.merge(prev, on=["constructorId", "year", "round"], how="left")
    bad = m[~np.isclose(m["con_points_before"], m["points"].astype(float).fillna(0))]
    refs = tables["constructors"].set_index("constructorId")["constructorRef"]
    found = set(zip(bad["year"], bad["constructorId"].map(refs)))
    assert found <= {(2007, "mclaren"), (2018, "force_india"), (2020, "racing_point")}, found


@pytest.mark.skipif(not RAW_PRESENT, reason="raw data not available")
def test_4_no_forbidden_columns_in_x(real):
    *_, X, y = real
    assert L.forbidden_columns(X, y) == []
    with pytest.raises(AssertionError):
        assert L.forbidden_columns(X.assign(points=0.0), y) == []   # the check really detects one


@pytest.mark.skipif(not RAW_PRESENT, reason="raw data not available")
def test_5_rows_identical_when_built_on_data_up_to_r(real):
    tables, rc, meta, X, y = real
    ids = [race_id(meta, 2023, 1), race_id(meta, 2012, 10), race_id(meta, 2021, 11),
           int(meta.sort_values(["year", "round"])["raceId"].iloc[-1])]
    out = L.prefix_invariance_check(tables, rc, ids, full=(meta, X, y))
    assert out["identical"].all(), out


@pytest.mark.skipif(not RAW_PRESENT, reason="raw data not available")
def test_6_split_disjoint_complete_and_ordered(real):
    meta = real[2]
    out = L.split_check(meta)
    assert out["disjoint"] and out["races_disjoint"] and out["complete"] and out["ordered"], out
    assert out["ranges"]["train"][1] == config.TRAIN_END_YEAR
    assert out["ranges"]["test"][0] == config.TEST_START_YEAR
