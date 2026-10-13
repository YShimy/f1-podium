"""Tests for the evaluation harness (evaluation.py) and the baseline models (models.py)."""

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score

from f1podium import config
from f1podium import evaluation as E
from f1podium import features as F
from f1podium import models as M

RAW_PRESENT = all((config.RAW_DIR / f"{t}.csv").exists() for t in ("results", "races", "qualifying"))


def synthetic_table(seed: int = 0):
    """Meta/X/y with all 21 features. Validation and test rows have very different values from train."""
    rng = np.random.default_rng(seed)
    years = np.repeat([2000, 2001, 2020, 2021, 2022, 2023], 20)
    n = len(years)
    meta = pd.DataFrame({"raceId": np.repeat(np.arange(len(years) // 10), 10), "driverId": np.tile(np.arange(10), n // 10),
                         "year": years, "round": 1})
    meta["split"] = F.assign_split(meta["year"])
    X = pd.DataFrame(rng.normal(size=(n, len(F.FEATURES))), columns=F.FEATURES)
    X["grid_filled"] = meta["driverId"] + 1.0
    X.loc[meta["year"] >= 2020, F.FEATURES[2:]] += 100.0          # shift every non-grid feature outside train
    X.loc[::4, "quali_position"] = np.nan                           # some missing values to impute
    X["has_quali"] = X["quali_position"].notna().astype(float)
    y = pd.Series((meta["driverId"] < 3).astype(int), name="podium")
    return meta, X, y


# --- 1. preprocessing is fitted on train rows only ---------------------------------------------

def test_imputer_statistics_are_train_medians():
    meta, X, y = synthetic_table()
    model = E.fit_on_train(M.logistic_baseline(), meta, X, y)
    imputer = model.named_steps["prep"].named_transformers_["num"].named_steps["impute"]
    train = X.loc[E.split_rows(meta, "train")]
    np.testing.assert_allclose(imputer.statistics_, train.median().to_numpy())
    assert not np.allclose(imputer.statistics_, X.median().to_numpy())          # all-rows median is different
    scaler = model.named_steps["prep"].named_transformers_["num"].named_steps["scale"]
    np.testing.assert_allclose(scaler.mean_, train.fillna(train.median()).mean().to_numpy())


@pytest.mark.skipif(not RAW_PRESENT, reason="raw data not available")
def test_imputer_uses_train_window_on_real_data():
    from f1podium.clean import clean_results
    from f1podium.load import load_all
    t = load_all()
    rc, _ = clean_results(t["results"], t["races"], t["circuits"], t["status"])
    meta, X, y = F.build_modeling_table(t, rc)
    model = E.fit_on_train(M.logistic_baseline(), meta, X, y)
    stats = pd.Series(model.named_steps["prep"].named_transformers_["num"].named_steps["impute"].statistics_,
                      index=F.FEATURES)
    train = X.loc[(meta["year"] >= config.TRAIN_START_YEAR) & (meta["year"] <= config.TRAIN_END_YEAR)]
    pd.testing.assert_series_equal(stats, train.median(), check_names=False)
    assert stats["drv_races_before"] != X["drv_races_before"].median()  # including val/test would change it


# --- 2. threshold choice ----------------------------------------------------------------------------

def test_f1_optimal_threshold_on_a_toy_example():
    #  score:  0.9  0.8  0.7  0.6  0.5  0.4
    #  label:   1    1    0    1    0    0
    # t=0.9: P=1, R=1/3, F1=0.5 | t=0.8: P=1, R=2/3, F1=0.8 | t=0.7: P=2/3, R=2/3, F1=0.667
    # t=0.6: P=3/4, R=1, F1=0.857 (best) | t=0.5: P=3/5, R=1, F1=0.75 | t=0.4: P=1/2, R=1, F1=0.667
    y = [1, 1, 0, 1, 0, 0]
    s = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4]
    t, f1 = E.f1_optimal_threshold(y, s)
    assert t == 0.6 and f1 == pytest.approx(6 / 7)
    assert E.threshold_metrics(y, s, t)["f1"] == pytest.approx(6 / 7)


def test_f1_threshold_ties_go_to_the_higher_threshold():
    # t=0.9 -> P=1, R=1/2, F1=2/3 ; t=0.5 -> P=1/2, R=1, F1=2/3 (tie) ; the higher one wins.
    t, f1 = E.f1_optimal_threshold([1, 0, 0, 1], [0.9, 0.8, 0.7, 0.5])
    assert t == 0.9 and f1 == pytest.approx(2 / 3)


# --- 3. the test split is locked --------------------------------------------------------------------

def test_split_rows_refuses_test():
    meta, *_ = synthetic_table()
    with pytest.raises(PermissionError):
        E.split_rows(meta, "test")


def test_final_test_report_refuses_to_run_twice(tmp_path):
    meta, X, y = synthetic_table()
    model = E.fit_on_train(M.GridBaseline(), meta, X, y)
    rec = E.final_test_report("grid", model, meta, X, y, threshold=-3.0, results_dir=tmp_path)
    assert rec["n"] == int((meta["split"] == "test").sum())
    with pytest.raises(RuntimeError, match="already evaluated"):
        E.final_test_report("grid", model, meta, X, y, threshold=-3.0, results_dir=tmp_path)
    E.final_test_report("other", model, meta, X, y, threshold=-3.0, results_dir=tmp_path)   # new name is fine
    E.final_test_report("grid", model, meta, X, y, threshold=-3.0, results_dir=tmp_path, force=True)
    log = pd.read_csv(tmp_path / E.FINAL_TEST_FILE)
    assert log["model"].tolist() == ["grid", "other", "grid"] and log["forced"].tolist() == [False, False, True]


# --- 4. race-level bootstrap -------------------------------------------------------------------------

def test_bootstrap_resamples_whole_races():
    sizes = {10: 3, 20: 5, 30: 2, 40: 7}
    race_ids = np.concatenate([[r] * k for r, k in sizes.items()])
    for idx in E.race_bootstrap_indices(race_ids, B=200, seed=1):
        picked = pd.Series(race_ids[idx]).value_counts()
        for race, count in picked.items():
            assert count % sizes[race] == 0                  # every pick brings all rows of that race
        # each race contributes its rows as complete blocks of positions
        for race in picked.index:
            positions = pd.Series(idx[race_ids[idx] == race]).value_counts()
            assert set(positions.index) == set(np.flatnonzero(race_ids == race))
            assert positions.nunique() == 1                  # every row of the race appears equally often
        assert sum(c // sizes[r] for r, c in picked.items()) == len(sizes)   # as many race draws as races


def test_bootstrap_ci_is_reproducible_and_contains_the_estimate():
    rng = np.random.default_rng(0)
    race_ids = np.repeat(np.arange(60), 20)
    y = np.tile((np.arange(20) < 3).astype(int), 60)
    s = -np.tile(np.arange(20), 60) + rng.normal(scale=4, size=len(y))
    a = E.race_bootstrap_ci(y, s, race_ids, B=300, seed=42)
    b = E.race_bootstrap_ci(y, s, race_ids, B=300, seed=42)
    assert a == b
    point = average_precision_score(y, s)
    assert a["pr_auc_ci"][0] <= point <= a["pr_auc_ci"][1]
    assert a["skipped"] == 0


# --- 5. baselines -------------------------------------------------------------------------------------

def test_quali_baseline_falls_back_to_grid():
    X = pd.DataFrame({"quali_position": [2.0, np.nan, 7.0], "has_quali": [1, 0, 1], "grid_filled": [5.0, 3.0, 1.0]})
    m = M.QualiBaseline().fit(X)
    np.testing.assert_array_equal(E.scores(m, X), [-2.0, -3.0, -7.0])
    np.testing.assert_array_equal(E.scores(M.GridBaseline().fit(X), X), [-5.0, -3.0, -1.0])


def test_scores_uses_predict_proba_for_logistic_regression():
    meta, X, y = synthetic_table()
    model = E.fit_on_train(M.logistic_baseline(), meta, X, y)
    s = E.scores(model, X)
    assert ((s >= 0) & (s <= 1)).all()
    assert M.LR_PARAMS["class_weight"] is None and M.LR_PARAMS["solver"] == "lbfgs"
