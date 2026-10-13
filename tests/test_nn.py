"""Tests for the shallow neural network (nn.py) and its use in the evaluation harness."""

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.metrics import average_precision_score

torch = pytest.importorskip("torch")

from f1podium import config  # noqa: E402
from f1podium import evaluation as E  # noqa: E402
from f1podium import features as F  # noqa: E402
from f1podium.nn import ShallowNNClassifier  # noqa: E402
from test_evaluation import synthetic_table  # noqa: E402


def noisy_table(seed=0):
    """Synthetic table where the target is weakly related to one feature (so early stopping has work to do)."""
    meta, X, y = synthetic_table(seed)
    rng = np.random.default_rng(seed)
    X = X.copy()
    signal = rng.normal(size=len(X))
    X["grid_filled"] = signal + 10
    y = pd.Series((signal + rng.normal(scale=1.5, size=len(X)) > 1.0).astype(int), name="podium")
    return meta, X, y


FAST = {"hidden_sizes": (8,), "max_epochs": 15, "patience": 4}


def test_two_runs_give_identical_predictions_and_history():
    meta, X, y = noisy_table()
    a = E.fit_on_train(ShallowNNClassifier(**FAST), meta, X, y)
    b = E.fit_on_train(ShallowNNClassifier(**FAST), meta, X, y)
    rows = E.split_rows(meta, "validation")
    np.testing.assert_array_equal(a.predict_proba(X.loc[rows]), b.predict_proba(X.loc[rows]))
    pd.testing.assert_frame_equal(a.history_, b.history_)
    assert a.best_epoch_ == b.best_epoch_


def test_different_seeds_differ():
    meta, X, y = noisy_table()
    rows = E.split_rows(meta, "validation")
    a = E.fit_on_train(ShallowNNClassifier(**FAST, seed=1), meta, X, y).predict_proba(X.loc[rows])
    b = E.fit_on_train(ShallowNNClassifier(**FAST, seed=2), meta, X, y).predict_proba(X.loc[rows])
    assert not np.array_equal(a, b)


def test_global_torch_settings_are_restored():
    meta, X, y = noisy_table()
    before = (torch.are_deterministic_algorithms_enabled(), torch.get_num_threads())
    m = E.fit_on_train(ShallowNNClassifier(**FAST), meta, X, y)
    m.predict_proba(X.head(5))
    assert (torch.are_deterministic_algorithms_enabled(), torch.get_num_threads()) == before


def test_preprocessing_is_fitted_on_train_rows_only():
    meta, X, y = noisy_table()
    m = E.fit_on_train(ShallowNNClassifier(**FAST), meta, X, y)
    num = m.prep_.named_transformers_["num"].named_steps
    train = X.loc[E.split_rows(meta, "train")]
    np.testing.assert_allclose(num["impute"].statistics_, train.median().to_numpy())
    filled = train.fillna(train.median())
    np.testing.assert_allclose(num["scale"].mean_, filled.mean().to_numpy())
    np.testing.assert_allclose(num["scale"].scale_, filled.std(ddof=0).to_numpy())


def test_early_stopping_restores_the_best_validation_epoch():
    meta, X, y = noisy_table()
    m = ShallowNNClassifier(hidden_sizes=(64,), lr=0.01, max_epochs=40, patience=3)
    E.fit_on_train(m, meta, X, y)
    h = m.history_
    assert len(h) < 40 and len(h) == m.best_epoch_ + 3                     # stopped after `patience` bad epochs
    assert h["val_pr_auc"].idxmax() + 1 == m.best_epoch_
    vrows = E.split_rows(meta, "validation")
    restored = average_precision_score(y.loc[vrows], m.predict_proba(X.loc[vrows])[:, 1])
    assert restored == pytest.approx(h["val_pr_auc"].max(), abs=1e-6)       # best weights, not the last ones


def test_fit_without_eval_set_runs_all_epochs():
    meta, X, y = noisy_table()
    rows = E.split_rows(meta, "train")
    m = ShallowNNClassifier(hidden_sizes=(8,), max_epochs=5, patience=1).fit(X.loc[rows], y.loc[rows])
    assert len(m.history_) == 5 and m.history_["val_pr_auc"].isna().all()


def test_sklearn_interface_and_probabilities():
    meta, X, y = noisy_table()
    m = clone(ShallowNNClassifier(**FAST))
    assert m.get_params()["hidden_sizes"] == (8,) and m.get_params()["seed"] == config.RANDOM_SEED
    with pytest.raises(AttributeError):
        m.predict_proba(X)                                                  # not fitted
    E.fit_on_train(m, meta, X, y)
    p = m.predict_proba(X.head(20))
    assert p.shape == (20, 2) and np.allclose(p.sum(axis=1), 1.0) and ((p >= 0) & (p <= 1)).all()
    assert list(m.classes_) == [0, 1]
    s = E.scores(m, X.head(20))                                             # harness scoring uses predict_proba
    np.testing.assert_array_equal(s, p[:, 1])


def test_harness_never_reads_test_rows_for_the_network(monkeypatch):
    meta, X, y = noisy_table()
    y = y.astype(float)
    y[meta["split"] == "test"] = np.nan
    X = X.copy()
    X.loc[meta["split"] == "test"] = np.nan
    requested = []
    real = E.split_rows
    monkeypatch.setattr(E, "split_rows", lambda m, s: (requested.append(s), real(m, s))[1])
    row, model, _ = E.evaluate_train_val("nn", ShallowNNClassifier(**FAST), meta, X, y)
    assert set(requested) <= {"train", "validation"}
    assert row["epochs_run"] == len(model.history_) and row["best_epoch"] == model.best_epoch_


def test_confusion_counts_are_consistent_with_precision_and_recall():
    y = np.array([1, 1, 1, 0, 0, 0, 0, 1])
    s = np.array([0.9, 0.8, 0.3, 0.7, 0.2, 0.1, 0.1, 0.6])
    m = E.threshold_metrics(y, s, 0.5)
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (3, 1, 1, 3)
    assert m["precision"] == pytest.approx(3 / 4) and m["recall"] == pytest.approx(3 / 4)
    assert m["tp"] + m["fp"] + m["fn"] + m["tn"] == len(y)


def test_nn_grid_is_small_and_one_to_two_hidden_layers():
    variants = E.param_variants(config.NN_GRID)
    assert len(variants) == 3 * 2 * 2 * 2
    assert all(1 <= len(v["hidden_sizes"]) <= 2 for v in variants)
    assert config.NN_FIXED["max_epochs"] == 100 and config.NN_FIXED["patience"] == 10
