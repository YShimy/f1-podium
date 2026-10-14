"""Tests for the section 13 experiments (experiments.py) and the options they add to the harness."""

import numpy as np
import pandas as pd
import pytest

from f1podium import config
from f1podium import evaluation as E
from f1podium import experiments as X_
from f1podium import features as F
from f1podium import models as M
from test_evaluation import synthetic_table


def windowed_table():
    """synthetic_table with train seasons spread over 1990-2019, so a window cut has something to remove."""
    meta, X, y = synthetic_table()
    meta = meta.copy()
    train = meta["split"] == F.SPLIT_TRAIN
    meta.loc[train, "year"] = np.resize([1990, 1995, 2000, 2005, 2010, 2019], int(train.sum()))
    return meta, X, y


# --- training window ------------------------------------------------------------------------------

@pytest.mark.parametrize("start", [1996, 2003, 2014])
def test_window_filter_excludes_earlier_seasons(start):
    meta, X, y = windowed_table()
    rows = E.split_rows(meta, "train", train_start=start)
    assert len(rows) > 0 and meta.loc[rows, "year"].min() >= start
    assert meta.loc[rows, "year"].max() <= config.TRAIN_END_YEAR
    expected = meta.index[(meta["split"] == F.SPLIT_TRAIN) & (meta["year"] >= start)]
    pd.testing.assert_index_equal(rows, expected)
    # the validation rows do not depend on the window
    pd.testing.assert_index_equal(E.split_rows(meta, "validation"), E.split_rows(meta, "validation", train_start=start))


def test_window_model_is_fitted_on_window_rows_only():
    meta, X, y = windowed_table()
    model = E.fit_on_train(M.logistic_baseline(), meta, X, y, train_start=2003)
    imputer = model.named_steps["prep"].named_transformers_["num"].named_steps["impute"]
    window = X.loc[(meta["split"] == F.SPLIT_TRAIN) & (meta["year"] >= 2003)]
    np.testing.assert_allclose(imputer.statistics_, window.median().to_numpy())
    assert not np.allclose(imputer.statistics_, X.loc[E.split_rows(meta, "train")].median().to_numpy())


def test_run_window_reports_shrinking_train_sets_and_same_validation():
    meta, X, y = windowed_table()
    table, scores = X_.run_window(meta, X, y, starts=(1950, 2003))
    assert table.loc[1950, "train_n"] > table.loc[2003, "train_n"]
    assert table.loc[1950, "val_n"] == table.loc[2003, "val_n"]
    assert set(scores) == {1950, 2003}


def test_default_window_is_unchanged():
    meta, X, y = windowed_table()
    pd.testing.assert_index_equal(E.split_rows(meta, "train"),
                                  E.split_rows(meta, "train", train_start=config.TRAIN_START_YEAR))


# --- experiment A: preprocessor fitted on train vs train+val ------------------------------------

def test_leakage_variant_uses_train_and_validation_statistics_and_clean_variant_does_not():
    meta, X, y = synthetic_table()
    train, val = E.split_rows(meta, "train"), E.split_rows(meta, "validation")
    leaky = X_.lr_with_prep_fitted_on(meta, X, y, "train+val")
    clean = X_.lr_with_prep_fitted_on(meta, X, y, "train")
    stats = lambda m: m.named_steps["prep"].named_transformers_["num"].named_steps["impute"].statistics_
    np.testing.assert_allclose(stats(leaky), X.loc[train.append(val)].median().to_numpy())
    np.testing.assert_allclose(stats(clean), X.loc[train].median().to_numpy())
    assert not np.allclose(stats(leaky), stats(clean))           # validation rows (shifted by +100) changed them
    # the test rows never enter: including them would change the medians again
    assert not np.allclose(stats(leaky), X.median().to_numpy())
    # the classifier itself is fitted on train rows only in both variants
    assert leaky.named_steps["clf"].n_features_in_ == len(F.FEATURES)


def test_clean_variant_equals_the_normal_pipeline():
    meta, X, y = synthetic_table()
    clean = X_.lr_with_prep_fitted_on(meta, X, y, "train")
    normal = E.fit_on_train(M.logistic_baseline(), meta, X, y)
    rows = E.split_rows(meta, "validation")
    np.testing.assert_allclose(clean.predict_proba(X.loc[rows]), normal.predict_proba(X.loc[rows]))


def test_prep_rows_refuses_test():
    meta, _, _ = synthetic_table()
    with pytest.raises(ValueError):
        X_.prep_fit_index(meta, "all")


# --- experiment B: preprocessing order --------------------------------------------------------------

def test_scale_then_impute_has_no_missing_values_and_reverses_the_steps():
    meta, X, y = synthetic_table()
    prep = M.build_preprocessor(scale=True, order="scale_impute")
    out = prep.fit_transform(X.loc[E.split_rows(meta, "train")])
    assert list(out.columns) == F.FEATURES and not out.isna().any().any()
    steps = [n for n, _ in prep.transformers[0][1].steps]
    assert steps == ["scale", "impute"]
    # the observed values are standardised with the observed-only mean, so the filled column is not mean 0
    q = out["quali_position"]
    assert abs(q.mean()) > 1e-6 or abs(q.std(ddof=0) - 1) > 1e-6


def test_unknown_order_is_refused():
    with pytest.raises(ValueError):
        M.build_preprocessor(scale=True, order="impute_only")


def test_nn_accepts_the_preprocessing_order():
    torch = pytest.importorskip("torch")  # noqa: F841
    from f1podium.nn import ShallowNNClassifier
    meta, X, y = synthetic_table()
    model = E.fit_on_train(ShallowNNClassifier(hidden_sizes=(4,), max_epochs=2, patience=1,
                                               prep_order="scale_impute"), meta, X, y)
    assert [n for n, _ in model.prep_.transformers_[0][1].steps] == ["scale", "impute"]


# --- ablation ---------------------------------------------------------------------------------------

def test_feature_groups_cover_every_feature_exactly_once():
    flat = [c for cols in config.FEATURE_GROUPS.values() for c in cols]
    assert sorted(flat) == sorted(F.FEATURES) and len(flat) == len(set(flat))


@pytest.mark.parametrize("group", list(config.FEATURE_GROUPS))
def test_each_ablation_removes_exactly_its_group(group):
    meta, X, y = synthetic_table()
    cols = X_.ablation_variants()[f"without {group}"]
    model = E.fit_on_train(M.logistic_baseline(features=cols), meta, X, y)
    used = list(model.named_steps["prep"].get_feature_names_out())
    assert set(F.FEATURES) - set(used) == set(config.FEATURE_GROUPS[group])
    assert len(used) == len(F.FEATURES) - len(config.FEATURE_GROUPS[group])
    assert model.named_steps["clf"].n_features_in_ == len(used)


def test_cumulative_baselines_use_only_grid_and_qualifying():
    v = X_.ablation_variants()
    assert set(v["grid only"]) == set(config.FEATURE_GROUPS["grid"])
    assert set(v["grid + qualifying only"]) == set(config.FEATURE_GROUPS["grid"] + config.FEATURE_GROUPS["qualifying"])
    assert v["full model"] == F.FEATURES


def test_ablation_refuses_overlapping_groups():
    bad = {**config.FEATURE_GROUPS, "home": ("is_home", "grid_filled")}
    with pytest.raises(ValueError):
        X_.ablation_variants(bad)
