"""Tests for the explainability helpers (xai.py)."""

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin

from f1podium import config
from f1podium import evaluation as E
from f1podium import features as F
from f1podium import models as M
from f1podium import xai
from test_evaluation import synthetic_table


# --- group permutation shuffles exactly the group's columns -------------------------------------------

class _Spy(ClassifierMixin, BaseEstimator):
    """Records every frame it scores; the score is a deterministic function of the row."""

    def __init__(self):
        self.seen = []

    def predict_proba(self, X):
        self.seen.append(X.copy())
        z = X.to_numpy().sum(axis=1)
        p = 1 / (1 + np.exp(-(z - z.mean()) / (z.std() + 1e-9)))
        return np.column_stack([1 - p, p])


def coded_frame(n=40):
    """Cell (r, j) = 1000 * r + j, so the source row of any value can be read back from the value."""
    return pd.DataFrame(1000.0 * np.arange(n)[:, None] + np.arange(len(F.FEATURES))[None, :], columns=F.FEATURES)


def test_group_permutation_shuffles_exactly_the_group_columns_jointly():
    X = coded_frame()
    y = np.resize([1, 0, 0, 0], len(X))
    spy = _Spy()
    n_repeats = 3
    table = xai.group_permutation_importance(spy, X, y, n_repeats=n_repeats)
    assert set(table.index) == set(config.FEATURE_GROUPS)
    original, calls = spy.seen[0], spy.seen[1:]
    pd.testing.assert_frame_equal(original, X)                       # the baseline call sees the untouched data
    assert len(calls) == len(config.FEATURE_GROUPS) * n_repeats
    for k, Xp in enumerate(calls):
        group = list(config.FEATURE_GROUPS)[k // n_repeats]
        cols = list(config.FEATURE_GROUPS[group])
        others = [c for c in F.FEATURES if c not in cols]
        pd.testing.assert_frame_equal(Xp[others], X[others])         # nothing outside the group moves
        source = (Xp[cols].to_numpy() // 1000).astype(int)           # row each value came from
        assert (source == source[:, [0]]).all()                      # one shared permutation across the group
        assert sorted(source[:, 0]) == list(range(len(X)))           # a true permutation of the rows
        assert (source[:, 0] != np.arange(len(X))).any()             # and the rows really moved
        np.testing.assert_array_equal(Xp[cols].to_numpy() % 1000, X[cols].to_numpy() % 1000)  # columns kept


def test_group_permutation_is_reproducible():
    meta, X, y = synthetic_table()
    model = E.fit_on_train(M.logistic_baseline(), meta, X, y)
    rows = E.split_rows(meta, "validation")
    a = xai.group_permutation_importance(model, X.loc[rows], y.loc[rows], n_repeats=4)
    b = xai.group_permutation_importance(model, X.loc[rows], y.loc[rows], n_repeats=4)
    pd.testing.assert_frame_equal(a, b)


# --- SHAP values + base value reproduce the model's log-odds ----------------------------------------------

@pytest.fixture(scope="module")
def fitted():
    meta, X, y = synthetic_table()
    lr = E.fit_on_train(M.logistic_baseline(), meta, X, y)
    hgb = E.fit_on_train(M.hist_gradient_boosting(max_iter=20, max_depth=3), meta, X, y)
    return meta, X, y, lr, hgb


def test_linear_shap_reproduces_the_log_odds(fitted):
    meta, X, y, lr, _ = fitted
    train, val = E.split_rows(meta, "train"), E.split_rows(meta, "validation")
    expl = xai.linear_shap(lr, X.loc[train], X.loc[val])
    logit = lr.named_steps["clf"].decision_function(xai.transformed_frame(lr, X.loc[val]))
    np.testing.assert_allclose(expl.values.sum(axis=1) + expl.base_values, logit, atol=1e-6)
    assert expl.feature_names == F.FEATURES
    np.testing.assert_array_equal(expl.display_data, X.loc[val].to_numpy())   # raw values for display
    # base value = model output at the mean of the WHOLE background (no 100-row subsample)
    bg = xai.transformed_frame(lr, X.loc[train]).mean()
    clf = lr.named_steps["clf"]
    np.testing.assert_allclose(expl.base_values, float(clf.intercept_[0] + clf.coef_[0] @ bg.to_numpy()), atol=1e-9)


def test_tree_shap_reproduces_the_log_odds(fitted):
    meta, X, y, _, hgb = fitted
    train, val = E.split_rows(meta, "train"), E.split_rows(meta, "validation")
    expl = xai.tree_shap(hgb, X.loc[val])
    raw = hgb.named_steps["clf"].decision_function(xai.transformed_frame(hgb, X.loc[val]))
    np.testing.assert_allclose(expl.values.sum(axis=1) + expl.base_values, raw, atol=1e-6)
    assert expl.feature_names == F.FEATURES


def test_tree_shap_adds_up_when_values_sit_on_split_thresholds():
    # integer-valued features with many distinct values: HistGB then puts thresholds on data values (the case of
    # the real model); path-dependent SHAP must still add up exactly
    meta, X, y = synthetic_table()
    X = X.copy()
    X[F.FEATURES[2:]] = np.round(X[F.FEATURES[2:]] * 3)
    hgb = E.fit_on_train(M.hist_gradient_boosting(max_iter=50, max_depth=3), meta, X, y)
    val = E.split_rows(meta, "validation")
    expl = xai.tree_shap(hgb, X.loc[val])
    raw = hgb.named_steps["clf"].decision_function(xai.transformed_frame(hgb, X.loc[val]))
    np.testing.assert_allclose(expl.values.sum(axis=1) + expl.base_values, raw, atol=1e-6)


def test_transformed_frame_keeps_original_names(fitted):
    meta, X, y, lr, hgb = fitted
    for model in (lr, hgb):
        assert list(xai.transformed_frame(model, X.head()).columns) == F.FEATURES
    assert xai.original_name("num__grid_filled") == "grid_filled" and xai.original_name("is_home") == "is_home"


# --- local rows and LIME --------------------------------------------------------------------------------------

def test_pick_local_rows_returns_the_stated_kinds():
    s = pd.Series([0.9, 0.8, 0.1, 0.05, 0.7, 0.2], index=list("abcdef"))
    y = pd.Series([1, 1, 1, 0, 0, 0], index=s.index)
    rows = xai.pick_local_rows(y, s, threshold=0.5)
    assert rows == {"confident correct podium": "a", "false negative": "c", "false positive": "e"}
    with pytest.raises(ValueError):
        xai.pick_local_rows(y[["a", "d"]], s[["a", "d"]], threshold=0.5)


def test_lime_is_deterministic_for_a_fixed_seed(fitted):
    meta, X, y, lr, _ = fitted
    train, val = E.split_rows(meta, "train"), E.split_rows(meta, "validation")
    row = X.loc[[val[0]]]
    a = xai.lime_explanation(lr, X.loc[train], row, num_samples=500).as_list()
    b = xai.lime_explanation(lr, X.loc[train], row, num_samples=500).as_list()
    assert a == b and len(a) == 10


def test_feature_permutation_importance_covers_every_feature(fitted):
    meta, X, y, lr, _ = fitted
    val = E.split_rows(meta, "validation")
    t = xai.feature_permutation_importance(lr, X.loc[val], y.loc[val], n_repeats=3)
    assert set(t.index) == set(F.FEATURES) and (t["importance_std"] >= 0).all()


def test_rank_agreement():
    a = pd.Series([5, 4, 3, 2, 1, 0.5], index=list("abcdef"))
    out = xai.rank_agreement(a, a * 2, top=3)
    assert out["spearman"] == pytest.approx(1.0) and out["top3_overlap"] == 3
