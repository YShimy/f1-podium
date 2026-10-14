"""Explainability helpers (section 16): permutation importance, SHAP and LIME for the fitted pipelines.

Everything here explains an ALREADY FITTED pipeline (median imputer [+ scaler] + classifier, fitted on train rows)
on the rows it is given; nothing is refitted. The notebook passes validation rows only. Explanations describe
predictive association and how the model relies on its inputs, not causal effects.
"""

import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score

from f1podium import config
from f1podium import evaluation as ev
from f1podium.features import FEATURES

with warnings.catch_warnings():
    # shap 0.52 calls Colormap.set_bad / set_over / set_under when its colour module is imported; matplotlib 3.11
    # marks those calls as pending deprecation. External and harmless, so only these messages, raised from shap's
    # own modules, are ignored, and only during this import.
    warnings.filterwarnings("ignore", message=r"The set_(bad|over|under) function will be deprecated",
                            category=PendingDeprecationWarning, module=r"shap\.")
    import shap  # noqa: E402

from lime.lime_tabular import LimeTabularExplainer  # noqa: E402

SCORING = "average_precision"
LOCAL_REASONS = ("confident correct podium", "false negative", "false positive")


def original_name(name: str) -> str:
    """Map a transformed column name back to the original feature (drops a ColumnTransformer 'num__' prefix)."""
    return name.split("__", 1)[1] if "__" in name else name


def transformed_frame(pipeline, X: pd.DataFrame) -> pd.DataFrame:
    """The classifier's input: X passed through the fitted preprocessing step, with original feature names."""
    t = pipeline.named_steps["prep"].transform(X)
    t = pd.DataFrame(t, index=X.index, columns=[original_name(c) for c in pipeline.named_steps["prep"].get_feature_names_out()])
    if list(t.columns) != list(FEATURES):
        raise ValueError(f"transformed columns {list(t.columns)} do not match FEATURES")
    return t


def _summary(name_to_drops: dict, baseline: float) -> pd.DataFrame:
    out = pd.DataFrame({"importance_mean": {k: float(np.mean(v)) for k, v in name_to_drops.items()},
                        "importance_std": {k: float(np.std(v)) for k, v in name_to_drops.items()}})
    out.attrs["baseline_pr_auc"] = baseline
    return out.sort_values("importance_mean", ascending=False)


def feature_permutation_importance(pipeline, X: pd.DataFrame, y, n_repeats: int = 30,
                                   seed: int = config.RANDOM_SEED) -> pd.DataFrame:
    """Drop in PR-AUC when one raw column is shuffled (sklearn permutation_importance on the whole pipeline)."""
    r = permutation_importance(pipeline, X, y, scoring=SCORING, n_repeats=n_repeats, random_state=seed)
    baseline = average_precision_score(y, ev.scores(pipeline, X))
    return _summary({c: r.importances[i] for i, c in enumerate(X.columns)}, baseline)


def group_permutation_importance(pipeline, X: pd.DataFrame, y, groups: dict = config.FEATURE_GROUPS,
                                 n_repeats: int = 30, seed: int = config.RANDOM_SEED) -> pd.DataFrame:
    """Drop in PR-AUC when ALL columns of a group are shuffled together (one row permutation per repeat).

    Shuffling the group jointly keeps each row's values within the group consistent with each other (for example
    quali_position and has_quali), while breaking their link with the other columns and with the target.
    """
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    baseline = average_precision_score(y, ev.scores(pipeline, X))
    drops = {}
    for g, cols in groups.items():
        cols = list(cols)
        drops[g] = []
        for _ in range(n_repeats):
            perm = rng.permutation(len(X))
            Xp = X.copy()
            Xp[cols] = X[cols].to_numpy()[perm]
            drops[g].append(baseline - average_precision_score(y, ev.scores(pipeline, Xp)))
    return _summary(drops, baseline)


def _explanation(values, base_values, data: pd.DataFrame, display: pd.DataFrame) -> shap.Explanation:
    return shap.Explanation(values=np.asarray(values), base_values=np.asarray(base_values, dtype=float),
                            data=data.to_numpy(), display_data=display.to_numpy(), feature_names=list(data.columns))


def linear_shap(pipeline, X_background: pd.DataFrame, X_explain: pd.DataFrame) -> shap.Explanation:
    """SHAP values (log-odds) of the logistic-regression step, background = ALL rows of X_background (transformed).

    An explicit Independent masker with max_samples = all rows is used; passing the frame directly would make
    shap subsample 100 background rows. display_data holds the raw (untransformed) inputs for readable plots.
    """
    bg = transformed_frame(pipeline, X_background)
    explainer = shap.LinearExplainer(pipeline.named_steps["clf"], shap.maskers.Independent(bg, max_samples=len(bg)))
    t = transformed_frame(pipeline, X_explain)
    e = explainer(t)
    return _explanation(e.values, e.base_values, t, X_explain)


def tree_shap(pipeline, X_background: pd.DataFrame, X_explain: pd.DataFrame, n_background: int = 500,
              seed: int = config.RANDOM_SEED) -> shap.Explanation:
    """Interventional TreeExplainer SHAP values (raw log-odds) of the HistGradientBoosting step.

    Background: a seeded sample of n_background rows of X_background, passed through the pipeline's imputer.
    """
    bg = transformed_frame(pipeline, X_background.sample(n=min(n_background, len(X_background)), random_state=seed))
    explainer = shap.TreeExplainer(pipeline.named_steps["clf"], data=bg, feature_perturbation="interventional",
                                   model_output="raw")
    t = transformed_frame(pipeline, X_explain)
    values = explainer.shap_values(t)
    return _explanation(values, np.full(len(t), float(np.ravel(explainer.expected_value)[0])), t, X_explain)


def mean_abs_shap(expl: shap.Explanation) -> pd.Series:
    return pd.Series(np.abs(expl.values).mean(axis=0), index=expl.feature_names,
                     name="mean |SHAP|").sort_values(ascending=False)


def lime_explanation(pipeline, X_train: pd.DataFrame, x_row: pd.DataFrame, seed: int = config.RANDOM_SEED,
                     num_features: int = 10, num_samples: int = 5000):
    """LIME tabular explanation of one row (1-row DataFrame) on the transformed train matrix.

    predict_fn is the fitted classifier's predict_proba on transformed inputs (wrapped in a DataFrame so the
    classifier sees the feature names it was fitted with).
    """
    train_t = transformed_frame(pipeline, X_train)
    row_t = transformed_frame(pipeline, x_row)
    clf = pipeline.named_steps["clf"]
    explainer = LimeTabularExplainer(train_t.to_numpy(), feature_names=list(train_t.columns),
                                     class_names=["no podium", "podium"], mode="classification",
                                     discretize_continuous=True, random_state=seed)
    predict = lambda Z: clf.predict_proba(pd.DataFrame(Z, columns=train_t.columns))  # noqa: E731
    return explainer.explain_instance(row_t.iloc[0].to_numpy(), predict, num_features=num_features,
                                      num_samples=num_samples)


def pick_local_rows(y: pd.Series, scores: pd.Series, threshold: float) -> dict:
    """Three rows to explain (ties: the first in index order).

    confident correct podium = the podium row with the highest score (a true positive);
    false negative = the podium row with the lowest score below the threshold (the most confident miss);
    false positive = the non-podium row with the highest score at or above the threshold.
    """
    y = pd.Series(np.asarray(y), index=scores.index)
    pred = scores >= threshold
    tp, fn, fp = scores[(y == 1) & pred], scores[(y == 1) & ~pred], scores[(y == 0) & pred]
    if tp.empty or fn.empty or fp.empty:
        raise ValueError("need at least one true positive, false negative and false positive")
    return dict(zip(LOCAL_REASONS, (tp.idxmax(), fn.idxmin(), fp.idxmax())))


def rank_agreement(a: pd.Series, b: pd.Series, top: int = 5) -> dict:
    """Spearman rank correlation of two importance vectors (same index) and the overlap of their top-k."""
    b = b.reindex(a.index)
    rho = spearmanr(a.to_numpy(), b.to_numpy()).statistic
    top_a, top_b = set(a.nlargest(top).index), set(b.nlargest(top).index)
    return {"spearman": float(rho), f"top{top}_overlap": len(top_a & top_b), f"top{top}_both": sorted(top_a & top_b)}
