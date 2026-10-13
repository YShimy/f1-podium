"""Evaluation harness.

Development uses the TRAIN and VALIDATION splits only: models are fitted on training rows
(config.TRAIN_START_YEAR..TRAIN_END_YEAR), and every choice (model, hyperparameters, decision
threshold) is made on validation. The test split is touched by exactly one function,
final_test_report, which records each run in results/final_test.csv and refuses to score the same
model name twice unless force=True.
"""

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

from f1podium import config
from f1podium.features import SPLIT_TEST, SPLIT_TRAIN, SPLIT_VAL

FINAL_TEST_FILE = "final_test.csv"


def split_rows(meta: pd.DataFrame, split: str) -> pd.Index:
    """Index of the train rows (inside the training window) or validation rows. Test is refused here."""
    if split == SPLIT_TEST:
        raise PermissionError("the test split is only scored by final_test_report")
    if split == SPLIT_TRAIN:
        mask = (meta["split"] == SPLIT_TRAIN) & meta["year"].between(config.TRAIN_START_YEAR, config.TRAIN_END_YEAR)
    elif split == SPLIT_VAL:
        mask = meta["split"] == SPLIT_VAL
    else:
        raise ValueError(f"unknown split {split!r}")
    return meta.index[mask]


def fit_on_train(model, meta: pd.DataFrame, X: pd.DataFrame, y: pd.Series):
    """Fit `model` (in place) on training rows only and return it."""
    rows = split_rows(meta, SPLIT_TRAIN)
    return model.fit(X.loc[rows], y.loc[rows])


def scores(model, X: pd.DataFrame) -> np.ndarray:
    """Ranking score per row (higher = more likely podium)."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return np.asarray(model.decision_function(X), dtype=float)


def ranking_metrics(y, s) -> dict:
    y = np.asarray(y)
    return {"n": len(y), "base_rate": float(y.mean()), "pr_auc": float(average_precision_score(y, s)),
            "roc_auc": float(roc_auc_score(y, s))}


def f1_optimal_threshold(y, s) -> tuple[float, float]:
    """(threshold, F1) maximising F1 for the rule `score >= threshold`; ties go to the higher threshold."""
    precision, recall, thresholds = precision_recall_curve(y, s)
    p, r = precision[:-1], recall[:-1]          # the last point has no threshold
    with np.errstate(divide="ignore", invalid="ignore"):
        f1 = np.where(p + r > 0, 2 * p * r / (p + r), 0.0)
    best = np.flatnonzero(np.isclose(f1, f1.max(), rtol=0, atol=1e-12))
    i = best[np.argmax(thresholds[best])]
    return float(thresholds[i]), float(f1[i])


def threshold_metrics(y, s, threshold: float) -> dict:
    y = np.asarray(y).astype(bool)
    pred = np.asarray(s) >= threshold
    tp = int((pred & y).sum())
    precision = tp / pred.sum() if pred.sum() else 0.0
    recall = tp / y.sum() if y.sum() else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "predicted_positive_share": float(pred.mean())}


def evaluate_on_validation(name: str, model, meta: pd.DataFrame, X: pd.DataFrame, y: pd.Series) -> tuple[dict, object]:
    """Fit on train, score validation, choose the F1-optimal threshold on validation. Returns (row, fitted model)."""
    fit_on_train(model, meta, X, y)
    rows = split_rows(meta, SPLIT_VAL)
    s = scores(model, X.loc[rows])
    threshold, _ = f1_optimal_threshold(y.loc[rows], s)
    row = {"model": name, **ranking_metrics(y.loc[rows], s), "threshold": threshold,
           **threshold_metrics(y.loc[rows], s, threshold)}
    row["pr_auc_over_base_rate"] = row["pr_auc"] / row["base_rate"]
    return row, model


def race_bootstrap_indices(race_ids, B: int = config.BOOTSTRAP_B, seed: int = config.RANDOM_SEED):
    """Yield B arrays of row positions, each built by resampling whole races with replacement."""
    race_ids = np.asarray(race_ids)
    unique, inverse = np.unique(race_ids, return_inverse=True)
    groups = [np.flatnonzero(inverse == k) for k in range(len(unique))]
    rng = np.random.default_rng(seed)
    for _ in range(B):
        picks = rng.integers(0, len(unique), size=len(unique))
        yield np.concatenate([groups[k] for k in picks])


def race_bootstrap_ci(y, s, race_ids, B: int = config.BOOTSTRAP_B, seed: int = config.RANDOM_SEED,
                      level: float = 0.95) -> dict:
    """Percentile CIs for PR-AUC and ROC-AUC from a race-level bootstrap.

    Rows of one race are correlated (one podium per three places), so whole races are resampled.
    Resamples with only one class are skipped and counted.
    """
    y, s = np.asarray(y), np.asarray(s)
    pr, roc, skipped = [], [], 0
    for idx in race_bootstrap_indices(race_ids, B, seed):
        yb = y[idx]
        if yb.min() == yb.max():
            skipped += 1
            continue
        pr.append(average_precision_score(yb, s[idx]))
        roc.append(roc_auc_score(yb, s[idx]))
    lo, hi = (1 - level) / 2 * 100, (1 + level) / 2 * 100
    return {"pr_auc_ci": tuple(np.percentile(pr, [lo, hi])), "roc_auc_ci": tuple(np.percentile(roc, [lo, hi])),
            "B": B, "skipped": skipped}


def final_test_report(model_name: str, model, meta: pd.DataFrame, X: pd.DataFrame, y: pd.Series,
                      threshold: float, results_dir: Path | None = None, force: bool = False,
                      src_commit: str | None = None) -> dict:
    """THE ONLY place the test split is scored. Appends one row to <results_dir>/final_test.csv.

    `model` must already be fitted (on train) and `threshold` frozen on validation. Raises if
    `model_name` already has a row, unless force=True (the new row is appended; the old one is kept).
    """
    path = Path(config.RESULTS_DIR if results_dir is None else results_dir) / FINAL_TEST_FILE
    previous = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=["model"])
    if (previous["model"] == model_name).any() and not force:
        raise RuntimeError(f"{model_name!r} was already evaluated on test ({path}); pass force=True to re-run")

    rows = meta.index[meta["split"] == SPLIT_TEST]
    yt, st = y.loc[rows], scores(model, X.loc[rows])
    ci = race_bootstrap_ci(yt, st, meta.loc[rows, "raceId"])
    rec = {"timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"), "model": model_name,
           "src_commit": src_commit, **ranking_metrics(yt, st),
           "pr_auc_ci_low": ci["pr_auc_ci"][0], "pr_auc_ci_high": ci["pr_auc_ci"][1],
           "roc_auc_ci_low": ci["roc_auc_ci"][0], "roc_auc_ci_high": ci["roc_auc_ci"][1],
           "bootstrap_B": ci["B"], "bootstrap_skipped": ci["skipped"], "threshold": threshold,
           **threshold_metrics(yt, st, threshold), "forced": bool(force)}
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([rec]).to_csv(path, mode="a", header=not path.exists(), index=False)
    return rec
