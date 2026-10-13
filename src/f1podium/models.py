"""Baseline models. Every model exposes fit(X, y) and either predict_proba or decision_function.

Preprocessing lives inside the sklearn Pipeline, so it is fitted only on the rows passed to fit
(evaluation.fit_on_train passes training rows only). No missing-value indicators are added: the
modeling table already has the flags (has_quali, is_round_1, *_races_before).
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from f1podium import config
from f1podium.features import FEATURES

LR_PARAMS = {"solver": "lbfgs", "penalty": "l2", "C": 1.0, "max_iter": 1000, "class_weight": None,
             "random_state": config.RANDOM_SEED}


def build_preprocessor(scale: bool, features: list[str] = FEATURES) -> ColumnTransformer:
    """Median imputation over all features, then (for scale-sensitive models) standardisation."""
    steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("scale", StandardScaler()))
    prep = ColumnTransformer([("num", Pipeline(steps), list(features))], remainder="drop",
                             verbose_feature_names_out=False)
    return prep.set_output(transform="pandas")


def logistic_baseline() -> Pipeline:
    """Median imputation + standardisation + L2 logistic regression with fixed default parameters."""
    return Pipeline([("prep", build_preprocessor(scale=True)), ("clf", LogisticRegression(**LR_PARAMS))])


class _RuleBaseline(ClassifierMixin, BaseEstimator):
    """A fixed scoring rule: nothing is learned. Higher score = more likely podium."""

    required: tuple[str, ...] = ()

    def fit(self, X: pd.DataFrame, y=None):
        missing = set(self.required) - set(X.columns)
        if missing:
            raise ValueError(f"missing columns: {sorted(missing)}")
        self.classes_ = np.array([0, 1])
        return self


class GridBaseline(_RuleBaseline):
    """score = -grid_filled (pole position scores highest)."""

    required = ("grid_filled",)

    def decision_function(self, X: pd.DataFrame) -> np.ndarray:
        return -X["grid_filled"].to_numpy(float)


class QualiBaseline(_RuleBaseline):
    """score = -quali_position where has_quali == 1; otherwise falls back to -grid_filled."""

    required = ("quali_position", "has_quali", "grid_filled")

    def decision_function(self, X: pd.DataFrame) -> np.ndarray:
        has_q = X["has_quali"].to_numpy() == 1
        return np.where(has_q, -X["quali_position"].to_numpy(float), -X["grid_filled"].to_numpy(float))


def baseline_models() -> dict:
    return {
        "naive: -grid": GridBaseline(),
        "naive: -quali (fallback -grid)": QualiBaseline(),
        "logistic regression": logistic_baseline(),
    }
