"""Small, reusable helpers for exploratory analysis. Descriptive only: nothing here is a model feature."""

import numpy as np
import pandas as pd

PODIUM_MAX_POSITION = 3


def with_podium(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with a boolean `podium` column: positionOrder <= 3 (missing -> False)."""
    out = df.copy()
    out["podium"] = out["positionOrder"].le(PODIUM_MAX_POSITION).fillna(False).astype(bool)
    return out


def rate_by(df: pd.DataFrame, by: str | list[str], flag: str) -> pd.DataFrame:
    """Share of rows where the boolean column `flag` is True, per group.

    Returns one row per group with n (rows), n_flag (rows where flag is True)
    and rate (n_flag / n).
    """
    if df[flag].isna().any():
        raise ValueError(f"{flag!r} has missing values; make it a plain boolean first")
    grouped = df.groupby(by)[flag]
    out = pd.DataFrame({"n": grouped.size(), "n_flag": grouped.sum().astype(int)})
    out["rate"] = out["n_flag"] / out["n"]
    return out


MISSING_BUCKET = "missing"


def rate_by_bins(x: pd.Series, y: pd.Series, bins=None, labels=None, q: int | None = None) -> pd.DataFrame:
    """Podium rate (mean of the 0/1 target y) per bucket of x; missing x values get their own bucket.

    Either `bins` (edges for pd.cut, right-closed) with optional `labels`, or `q` for quantile buckets
    (pd.qcut with duplicate edges dropped: a feature with many ties, e.g. mostly zeros, gets fewer buckets).
    Returns n, podiums and rate per bucket, in bucket order with "missing" last.
    """
    if (bins is None) == (q is None):
        raise ValueError("give exactly one of bins or q")
    y = pd.Series(np.asarray(y), index=x.index)
    cat = pd.cut(x, bins=bins, labels=labels, right=True) if bins is not None else pd.qcut(x, q=q, duplicates="drop")
    bucket = cat.astype(str).where(x.notna(), MISSING_BUCKET)
    order = [str(c) for c in cat.cat.categories if str(c) in set(bucket)]
    order += [MISSING_BUCKET] if x.isna().any() else []
    g = y.groupby(bucket)
    out = pd.DataFrame({"n": g.size(), "podiums": g.sum().astype(int)}).reindex(order)
    out["rate"] = out["podiums"] / out["n"]
    out.index.name = x.name
    return out


def univariate_pr_auc(X: pd.DataFrame, y) -> pd.DataFrame:
    """Each feature used ALONE as a ranking score: PR-AUC on its non-missing rows, in its better direction.

    Exploratory screening only. The direction ("higher" or "lower" values mean more podiums) is whichever gives
    the larger PR-AUC on the rows given; base_rate is the podium share on those same rows, so compare `lift`
    (PR-AUC / base rate) across features with different coverage.
    """
    from sklearn.metrics import average_precision_score

    y = pd.Series(np.asarray(y), index=X.index)
    rows = []
    for c in X.columns:
        ok = X[c].notna()
        yc, xc = y[ok].to_numpy(), X.loc[ok, c].to_numpy(dtype=float)
        if len(yc) == 0 or yc.min() == yc.max():
            rows.append({"feature": c, "coverage": ok.mean(), "n": int(ok.sum())})
            continue
        hi, lo = average_precision_score(yc, xc), average_precision_score(yc, -xc)
        rows.append({"feature": c, "coverage": ok.mean(), "n": int(ok.sum()), "base_rate": yc.mean(),
                     "pr_auc": max(hi, lo), "direction": "higher" if hi >= lo else "lower"})
    out = pd.DataFrame(rows).set_index("feature")
    out["lift"] = out["pr_auc"] / out["base_rate"]
    return out.sort_values("lift", ascending=False)
