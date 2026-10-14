"""Baseline models. Every model exposes fit(X, y) and either predict_proba or decision_function.

Preprocessing lives inside the sklearn Pipeline, so it is fitted only on the rows passed to fit
(evaluation.fit_on_train passes training rows only). No missing-value indicators are added: the
modeling table already has the flags (has_quali, is_round_1, *_races_before).
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from f1podium import config
from f1podium.features import FEATURES

LR_PARAMS = {"solver": "lbfgs", "penalty": "l2", "C": 1.0, "max_iter": 1000, "class_weight": None,
             "random_state": config.RANDOM_SEED}


PREP_ORDERS = ("impute_scale", "scale_impute")


def build_preprocessor(scale: bool, features: list[str] = FEATURES, order: str = "impute_scale") -> ColumnTransformer:
    """Median imputation over all features, then (for scale-sensitive models) standardisation.

    order="scale_impute" (experiment only) standardises first (StandardScaler ignores NaN when fitting
    and keeps it when transforming), then fills the gaps with the median of the scaled column.
    """
    if order not in PREP_ORDERS:
        raise ValueError(f"order must be one of {PREP_ORDERS}, got {order!r}")
    steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("scale", StandardScaler()))
        if order == "scale_impute":
            steps.reverse()
    prep = ColumnTransformer([("num", Pipeline(steps), list(features))], remainder="drop",
                             verbose_feature_names_out=False)
    return prep.set_output(transform="pandas")


def logistic_baseline(*, features: list[str] = FEATURES, prep_order: str = "impute_scale", **params) -> Pipeline:
    """Median imputation + standardisation + L2 logistic regression with fixed default parameters.

    `params` override LR_PARAMS (used only for the class_weight comparison). `features` (a subset of the
    columns, for the ablation) and `prep_order` (impute-then-scale vs scale-then-impute) are only
    changed by the experiments; the defaults are the frozen model."""
    return Pipeline([("prep", build_preprocessor(scale=True, features=features, order=prep_order)),
                     ("clf", LogisticRegression(**{**LR_PARAMS, **params}))])


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


def random_forest(**params) -> Pipeline:
    """Same median-imputation preprocessing as LR (no scaler: trees are scale-free) + random forest.

    Defaults (overridden by params): 200 trees, unlimited depth, min_samples_leaf=1, class_weight=None;
    seed 42, single-threaded: with several threads the per-tree probabilities are summed in a varying
    order, so predictions differ at the 1e-16 level between runs (a test checks bit-identical results).
    """
    clf = RandomForestClassifier(**{"n_estimators": 200, "max_depth": None, "min_samples_leaf": 1,
                                    "class_weight": None, "random_state": config.RANDOM_SEED, "n_jobs": 1, **params})
    return Pipeline([("prep", build_preprocessor(scale=False)), ("clf", clf)])


def hist_gradient_boosting(**params) -> Pipeline:
    """Same preprocessing as the other models (median imputation, no scaler) + HistGradientBoosting.

    HistGradientBoosting handles NaN natively, so the imputation is irrelevant to it; it is kept
    identical so all models see exactly the same inputs. Early stopping is off (max_iter is a grid
    parameter and nothing is split off the training rows); seed 42.
    """
    clf = HistGradientBoostingClassifier(**{"learning_rate": 0.1, "max_iter": 100, "max_depth": None,
                                            "l2_regularization": 0.0, "class_weight": None,
                                            "early_stopping": False, "random_state": config.RANDOM_SEED, **params})
    return Pipeline([("prep", build_preprocessor(scale=False)), ("clf", clf)])


def baseline_models() -> dict:
    return {
        "naive: -grid": GridBaseline(),
        "naive: -quali (fallback -grid)": QualiBaseline(),
        "logistic regression": logistic_baseline(),
    }
