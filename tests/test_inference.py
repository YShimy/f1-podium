"""Tests for single-record inference (inference.py)."""

import numpy as np
import pandas as pd
import pytest

from f1podium import config
from f1podium import evaluation as E
from f1podium import features as F
from f1podium import models as M
from f1podium.inference import InferenceBundle, predict_podium, prepare_record
from test_evaluation import synthetic_table

PARTIAL_DROP = ["quali_position", "drv_rank_before", "con_podium_rate_last5"]


@pytest.fixture(scope="module")
def fitted():
    meta, X, y = synthetic_table()
    pipe = E.fit_on_train(M.logistic_baseline(), meta, X, y)
    X = X.copy()
    X["has_quali"] = X["quali_position"].notna().astype(float)     # as in the modeling table
    rows = E.split_rows(meta, "validation")
    return InferenceBundle(pipeline=pipe, threshold=0.5, freeze_hash="abc"), X, rows


def a_row_with_quali(X, rows):
    return next(i for i in rows if pd.notna(X.loc[i, "quali_position"]))


# --- (a) complete record ------------------------------------------------------------------------------

@pytest.mark.parametrize("kind", ["series", "dict", "frame"])
def test_complete_record_equals_the_pipeline(fitted, kind):
    bundle, X, rows = fitted
    i = a_row_with_quali(X, rows)
    record = {"series": X.loc[i], "dict": X.loc[i].to_dict(), "frame": X.loc[[i]]}[kind]
    out = predict_podium(record, bundle)
    expected = bundle.pipeline.predict_proba(X.loc[[i]])[0, 1]
    assert out["probability"] == expected
    assert out["decision"] == (expected >= bundle.threshold) and out["threshold"] == 0.5
    assert out["missing_inputs"] == [] and out["derived"] == {"has_quali": 1} and out["freeze_hash"] == "abc"


# --- (b) partially missing record ----------------------------------------------------------------------

def test_partial_record_is_imputed_by_the_pipeline_and_reports_missing_inputs(fitted):
    bundle, X, rows = fitted
    i = a_row_with_quali(X, rows)
    record = X.loc[i].drop(PARTIAL_DROP + ["has_quali"]).to_dict()
    out = predict_podium(record, bundle)
    assert out["missing_inputs"] == [c for c in F.FEATURES if c in PARTIAL_DROP]
    assert out["derived"] == {"has_quali": 0}                          # quali_position missing -> flag 0
    manual = X.loc[[i]].copy()
    manual[PARTIAL_DROP] = np.nan
    manual["has_quali"] = 0.0
    assert out["probability"] == bundle.pipeline.predict_proba(manual)[0, 1]
    assert 0.0 <= out["probability"] <= 1.0


def test_none_counts_as_missing_and_an_empty_record_is_fully_imputed(fitted):
    bundle, X, rows = fitted
    rec = X.loc[rows[0]].to_dict()
    rec["drv_rank_before"] = None
    assert "drv_rank_before" in predict_podium(rec, bundle)["missing_inputs"]
    out = predict_podium({}, bundle)
    assert len(out["missing_inputs"]) == len(F.FEATURES) - 1 and out["derived"] == {"has_quali": 0}


def test_has_quali_that_contradicts_quali_position_is_rejected(fitted):
    bundle, X, rows = fitted
    rec = X.loc[a_row_with_quali(X, rows)].to_dict()
    rec["has_quali"] = 0.0
    with pytest.raises(ValueError, match="has_quali=0 contradicts quali_position"):
        predict_podium(rec, bundle)


# --- (c) invalid records ---------------------------------------------------------------------------------

@pytest.mark.parametrize("col", list(config.FORBIDDEN_CURRENT_RACE_COLUMNS) + ["podium"])
def test_post_race_and_target_columns_are_rejected(fitted, col):
    bundle, X, rows = fitted
    rec = {**X.loc[rows[0]].to_dict(), col: 1}
    with pytest.raises(ValueError, match="post-race or target columns") as err:
        predict_podium(rec, bundle)
    assert col in str(err.value)


def test_unknown_columns_and_bad_shapes_are_rejected(fitted):
    bundle, X, rows = fitted
    with pytest.raises(ValueError, match=r"unknown columns \['grid'\]"):
        predict_podium({**X.loc[rows[0]].to_dict(), "grid": 3}, bundle)
    with pytest.raises(ValueError, match="exactly 1 row"):
        predict_podium(X.loc[rows[:2]], bundle)
    with pytest.raises(ValueError, match="must be numeric"):
        predict_podium({"grid_filled": "pole"}, bundle)
    with pytest.raises(TypeError):
        predict_podium([1, 2, 3], bundle)


def test_bundle_requires_a_fitted_pipeline_with_the_frozen_features(fitted):
    from sklearn.exceptions import NotFittedError
    with pytest.raises(NotFittedError):
        InferenceBundle(pipeline=M.logistic_baseline(), threshold=0.5)
    meta, X, y = synthetic_table()
    other = E.fit_on_train(M.logistic_baseline(features=F.FEATURES[:5]), meta, X, y)
    with pytest.raises(ValueError, match="frozen features"):
        InferenceBundle(pipeline=other, threshold=0.5)


# --- probability range and determinism -----------------------------------------------------------------

def test_probability_is_in_unit_interval_for_extreme_and_random_records(fitted):
    bundle, X, rows = fitted
    rng = np.random.default_rng(0)
    for scale in (1.0, 1e3, 1e6):
        rec = {c: float(v) for c, v in zip(F.FEATURES, rng.normal(scale=scale, size=len(F.FEATURES)))}
        rec["has_quali"] = 1.0
        p = predict_podium(rec, bundle)["probability"]
        assert 0.0 <= p <= 1.0


def test_predictions_are_deterministic_and_do_not_modify_the_record(fitted):
    bundle, X, rows = fitted
    rec = X.loc[rows[0]].drop(PARTIAL_DROP).to_dict()
    before = dict(rec)
    a, b = predict_podium(rec, bundle), predict_podium(rec, bundle)
    assert a == b and rec == before
    assert prepare_record(rec)[0].equals(prepare_record(rec)[0])
