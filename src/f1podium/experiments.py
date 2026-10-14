"""Section 13 experiments: training window, preprocessing order, feature-group ablation.

All of them use VALIDATION only. Models are fitted on train rows, each variant gets its own F1-optimal
threshold on validation (the same protocol as evaluation.evaluate_train_val), and the test split is
never read: rows come from evaluation.split_rows, which refuses test. Nothing here is used to choose a
model for test; the results are recorded as decisions in the notebook.
"""

import numpy as np
import pandas as pd

from f1podium import config
from f1podium import evaluation as ev
from f1podium import models as mdl
from f1podium.features import FEATURES, SPLIT_TRAIN, SPLIT_VAL

PREP_FIT_ROWS = ("train", "train+val")


def run_window(meta: pd.DataFrame, X: pd.DataFrame, y: pd.Series,
               starts=config.TRAIN_START_CANDIDATES) -> tuple[pd.DataFrame, dict[int, np.ndarray]]:
    """Frozen logistic regression trained on seasons start..TRAIN_END_YEAR, for each start.

    The whole pipeline (imputer, scaler, classifier) is fitted on the window's train rows only.
    Returns (one row per start, validation scores per start). Validation rows are the same for all.
    """
    rows, val_scores = [], {}
    for start in starts:
        row, _, sv = ev.evaluate_train_val(f"train from {start}", mdl.logistic_baseline(), meta, X, y,
                                           train_start=start)
        row["train_start"] = start
        rows.append(row)
        val_scores[start] = sv
    return pd.DataFrame(rows).set_index("train_start"), val_scores


def prep_fit_index(meta: pd.DataFrame, which: str) -> pd.Index:
    """Rows the preprocessor is fitted on: "train" (correct) or "train+val" (leakage demo). Never test."""
    if which not in PREP_FIT_ROWS:
        raise ValueError(f"which must be one of {PREP_FIT_ROWS}, got {which!r}")
    rows = ev.split_rows(meta, SPLIT_TRAIN)
    if which == "train+val":
        rows = rows.append(ev.split_rows(meta, SPLIT_VAL))
    return rows


def lr_with_prep_fitted_on(meta: pd.DataFrame, X: pd.DataFrame, y: pd.Series, which: str):
    """Frozen logistic regression whose imputer + scaler are fitted on `which` rows; the classifier on train only.

    which="train" reproduces the normal pipeline. which="train+val" is the leakage demo: the imputer
    medians and scaler means/SDs then contain validation rows, the rows the model is judged on.
    """
    model = mdl.logistic_baseline()
    prep, clf = model.named_steps["prep"], model.named_steps["clf"]
    prep.fit(X.loc[prep_fit_index(meta, which)])
    train = ev.split_rows(meta, SPLIT_TRAIN)
    clf.fit(prep.transform(X.loc[train]), y.loc[train])
    return model


def run_prep_rows(meta: pd.DataFrame, X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, dict, dict]:
    """Experiment A. Returns (metrics per variant, validation scores per variant, fitted models)."""
    rows, val_scores, models = [], {}, {}
    for which in PREP_FIT_ROWS:
        name = f"prep fitted on {which}"
        models[name] = lr_with_prep_fitted_on(meta, X, y, which)
        row, _, val_scores[name] = ev.score_train_val(name, models[name], meta, X, y)
        rows.append(row)
    return pd.DataFrame(rows).set_index("model"), val_scores, models


def prep_statistics(model, columns) -> pd.DataFrame:
    """Imputer median and scaler mean / SD of a fitted logistic-regression pipeline, for `columns`."""
    steps = model.named_steps["prep"].named_transformers_["num"].named_steps
    names = list(model.named_steps["prep"].get_feature_names_out())
    pos = [names.index(c) for c in columns]
    return pd.DataFrame({"imputer median": steps["impute"].statistics_[pos],
                         "scaler mean": steps["scale"].mean_[pos],
                         "scaler SD": steps["scale"].scale_[pos]}, index=list(columns))


def run_prep_order(meta: pd.DataFrame, X: pd.DataFrame, y: pd.Series, nn_params: dict | None = None,
                   nn_fixed: dict | None = None) -> tuple[pd.DataFrame, dict, dict]:
    """Experiment B: impute->scale vs scale->impute for logistic regression and (if nn_params) the frozen NN.

    The NN keeps its own protocol (one fixed seed, early stopping on validation). Returns (metrics per
    variant, validation scores per variant, fitted models).
    """
    makers = {"logistic regression": lambda order: mdl.logistic_baseline(prep_order=order)}
    if nn_params is not None:
        from f1podium.nn import ShallowNNClassifier
        makers["neural network"] = lambda order: ShallowNNClassifier(**nn_params, **(nn_fixed or {}), prep_order=order)
    rows, val_scores, models = [], {}, {}
    for model_name, make in makers.items():
        for order in mdl.PREP_ORDERS:
            name = f"{model_name} | {order.replace('_', '→')}"
            row, models[name], val_scores[name] = ev.evaluate_train_val(name, make(order), meta, X, y)
            row.update({"model family": model_name, "order": order})
            rows.append(row)
    return pd.DataFrame(rows).set_index("model"), val_scores, models


def ablation_variants(groups: dict = config.FEATURE_GROUPS) -> dict[str, list[str]]:
    """Feature lists: the full model, the full model minus each group, and two cumulative baselines."""
    flat = [c for cols in groups.values() for c in cols]
    if sorted(flat) != sorted(FEATURES) or len(flat) != len(set(flat)):
        raise ValueError("FEATURE_GROUPS must contain every feature exactly once")
    variants = {"full model": list(FEATURES)}
    for g, cols in groups.items():
        variants[f"without {g}"] = [c for c in FEATURES if c not in cols]
    variants["grid only"] = [c for c in FEATURES if c in groups["grid"]]
    variants["grid + qualifying only"] = [c for c in FEATURES if c in groups["grid"] + groups["qualifying"]]
    return variants


def run_ablation(meta: pd.DataFrame, X: pd.DataFrame, y: pd.Series,
                 groups: dict = config.FEATURE_GROUPS) -> tuple[pd.DataFrame, dict, dict]:
    """Frozen logistic regression on each feature list of ablation_variants (refitted, own validation threshold).

    Returns (metrics per variant incl. n_features, validation scores per variant, fitted models).
    """
    rows, val_scores, models = [], {}, {}
    for name, cols in ablation_variants(groups).items():
        row, models[name], val_scores[name] = ev.evaluate_train_val(name, mdl.logistic_baseline(features=cols),
                                                                    meta, X, y)
        row["n_features"] = len(cols)
        rows.append(row)
    return pd.DataFrame(rows).set_index("model"), val_scores, models
