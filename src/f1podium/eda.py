"""Small, reusable helpers for exploratory analysis. Descriptive only: nothing here is a model feature."""

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
