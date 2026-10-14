"""Inference for one driver-race record with the frozen final model.

The model is the logistic-regression pipeline fitted on train rows only (the one scored in the test report),
with its decision threshold frozen on validation. A record holds the 21 pre-race features (features.FEATURES),
all known before lights out: grid, qualifying, static race context, and form / standings from EARLIER races.

Rules applied to every record:
- Post-race columns (config.FORBIDDEN_CURRENT_RACE_COLUMNS, e.g. positionOrder, points, laps, status) and the
  target itself are rejected: they are not known at the prediction cutoff.
- Unknown columns are rejected (a typo would otherwise be silently ignored).
- Missing features are allowed: they are set to NaN and filled by the pipeline's own imputer (training-row
  medians). No preprocessing happens outside the fitted pipeline.
- Derived flag: has_quali = 1 if quali_position is present, else 0 (exactly as features.build_modeling_table
  builds it). A supplied has_quali that contradicts quali_position is rejected. No other flag is derived:
  pit_lane_start cannot be recovered from grid_filled (grid 0 is already replaced there), and is_round_1 cannot be
  inferred from a missing rank (the rank may simply be unknown); if absent they are imputed like any feature.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.utils.validation import check_is_fitted

from f1podium import config
from f1podium.features import FEATURES

TARGET_COLUMNS = ("podium",)
FORBIDDEN_INPUTS = tuple(config.FORBIDDEN_CURRENT_RACE_COLUMNS) + TARGET_COLUMNS
DERIVED_FLAGS = {"has_quali": "quali_position"}       # flag -> the column whose presence it records


@dataclass(frozen=True)
class InferenceBundle:
    """Everything needed to score a record: the fitted pipeline and its frozen validation threshold."""

    pipeline: object
    threshold: float
    model_name: str = "logistic regression"
    freeze_hash: str | None = None
    features: tuple = tuple(FEATURES)

    def __post_init__(self):
        check_is_fitted(self.pipeline)
        used = list(self.pipeline.named_steps["prep"].get_feature_names_out())
        if used != list(self.features):
            raise ValueError(f"pipeline uses {used}, expected the frozen features {list(self.features)}")
        if not np.isfinite(self.threshold):
            raise ValueError("threshold must be a finite number")


def _as_series(record) -> pd.Series:
    if isinstance(record, pd.DataFrame):
        if len(record) != 1:
            raise ValueError(f"a DataFrame record must have exactly 1 row, got {len(record)}")
        return record.iloc[0]
    if isinstance(record, pd.Series):
        return record
    if isinstance(record, dict):
        return pd.Series(record, dtype=object)
    raise TypeError(f"record must be a dict, Series or 1-row DataFrame, got {type(record).__name__}")


def prepare_record(record, features=FEATURES) -> tuple[pd.DataFrame, list[str], dict]:
    """Validate a record and return (1-row frame in feature order, missing inputs, derived flags).

    Raises ValueError for post-race / target columns, unknown columns, non-numeric values, or a has_quali that
    contradicts quali_position. The input is not modified.
    """
    s = _as_series(record)
    cols = [str(c) for c in s.index]
    forbidden = [c for c in cols if c in FORBIDDEN_INPUTS]
    if forbidden:
        raise ValueError(f"record contains post-race or target columns {forbidden}: they are not known before the "
                         "race starts (prediction cutoff: after qualifying, before lights out)")
    unknown = [c for c in cols if c not in features]
    if unknown:
        raise ValueError(f"record contains unknown columns {unknown}; expected only the features {list(features)}")

    values = {}
    for c in features:
        v = s.get(c, np.nan)
        v = np.nan if v is None else v
        try:
            values[c] = float(v)
        except (TypeError, ValueError):
            raise ValueError(f"feature {c!r} must be numeric, got {v!r}") from None
    derived = {}
    for flag, source in DERIVED_FLAGS.items():
        implied = 0.0 if np.isnan(values[source]) else 1.0
        if not np.isnan(values[flag]) and values[flag] != implied:
            raise ValueError(f"{flag}={values[flag]:g} contradicts {source} "
                             f"({'missing' if implied == 0 else 'present'}); {flag} must be {implied:g}")
        values[flag] = implied
        derived[flag] = int(implied)
    missing = [c for c in features if c not in DERIVED_FLAGS and np.isnan(values[c])]
    return pd.DataFrame([values], columns=list(features)), missing, derived


def predict_podium(record, bundle: InferenceBundle) -> dict:
    """Podium probability and decision for one pre-race record (see the module docstring for the rules).

    Returns {"model", "probability", "threshold", "decision", "missing_inputs", "derived", "freeze_hash"}.
    decision is True when probability >= the frozen validation threshold (the same rule as in evaluation).
    """
    row, missing, derived = prepare_record(record, bundle.features)
    p = float(bundle.pipeline.predict_proba(row)[0, 1])
    return {"model": bundle.model_name, "probability": p, "threshold": float(bundle.threshold),
            "decision": bool(p >= bundle.threshold), "missing_inputs": missing, "derived": derived,
            "freeze_hash": bundle.freeze_hash}
