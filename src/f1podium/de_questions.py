"""Data-engineering questions: descriptive analyses of race outcomes.

These use race results (e.g. who finished on the podium) to *describe* the sport.
They are not model features, so the prediction cutoff (after qualifying, before the
race) does not apply to them.

Q1: which circuits convert a front-row start into a podium most often, and does the
ranking change between eras? Qualifying position and grid position are kept apart.
"""

from math import erfc, sqrt

import numpy as np
import pandas as pd

from f1podium import config
from f1podium.eda import PODIUM_MAX_POSITION

KEY = ["raceId", "driverId"]
FRONT_ROW_POSITIONS = (1, 2)  # grid = 0 is never front row
Z_95 = 1.959963984540054

ERA_OLD = "≤2013"
ERA_NEW = "≥2014"
ERAS = (ERA_OLD, ERA_NEW)

WINDOW_PRIMARY = "2003–2024"
WINDOW_ALL = "all years"
DEF_GRID = "grid"
DEF_QUALI = "quali"
# (window label, definition label, flag column, year window or None for all years)
SPECS = [
    (WINDOW_PRIMARY, DEF_GRID, "front_row_grid", config.DE_WINDOW),
    (WINDOW_PRIMARY, DEF_QUALI, "front_row_quali", config.DE_WINDOW),
    (WINDOW_ALL, DEF_GRID, "front_row_grid", None),
]


def wilson_interval(successes, n, z: float = Z_95):
    """Wilson score interval for a binomial proportion.

    Accepts scalars or arrays. Returns (low, high); both are NaN where n == 0.
    """
    s = np.asarray(successes, dtype=float)
    n_ = np.asarray(n, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = s / n_
        denom = 1 + z**2 / n_
        centre = (p + z**2 / (2 * n_)) / denom
        half = z * np.sqrt(p * (1 - p) / n_ + z**2 / (4 * n_**2)) / denom
    low = np.where(n_ > 0, np.clip(centre - half, 0, 1), np.nan)[()]
    high = np.where(n_ > 0, np.clip(centre + half, 0, 1), np.nan)[()]
    return low, high


def _require_unique(df: pd.DataFrame, cols: list[str], name: str) -> None:
    if df[cols].isna().any().any():
        raise AssertionError(f"{name}: null values in key {cols}")
    if df.duplicated(cols).any():
        raise AssertionError(f"{name}: not unique on {cols}")


def _require_rows(df: pd.DataFrame, expected: int, step: str) -> None:
    if len(df) != expected:
        raise AssertionError(f"row count changed after {step}: {expected} -> {len(df)}")


def build_front_row_table(
    results_clean: pd.DataFrame,
    races: pd.DataFrame,
    circuits: pd.DataFrame,
    qualifying: pd.DataFrame,
) -> pd.DataFrame:
    """One row per (raceId, driverId) of the cleaned results, with circuit and qualifying info.

    Joins: cleaned results -> races (year, round, circuitId) -> circuits (name)
    -> LEFT qualifying on (raceId, driverId). Raises if a key is not unique or if
    any join changes the row count.

    Columns added: year, round, circuitId, circuit, quali_position, podium
    (positionOrder <= 3), front_row_grid (grid in {1, 2}) and front_row_quali
    (quali position in {1, 2}). Missing values are never front row.
    """
    _require_unique(results_clean, KEY, "results_clean")
    _require_unique(races, ["raceId"], "races")
    _require_unique(circuits, ["circuitId"], "circuits")
    _require_unique(qualifying, KEY, "qualifying")

    n = len(results_clean)
    out = results_clean[KEY + ["grid", "positionOrder"]].copy()

    out = out.merge(races[["raceId", "year", "round", "circuitId"]], on="raceId", how="left", validate="many_to_one")
    _require_rows(out, n, "join to races")
    if out[["year", "round", "circuitId"]].isna().any().any():
        raise AssertionError("some results have a raceId that is not in races")

    out = out.merge(
        circuits[["circuitId", "name"]].rename(columns={"name": "circuit"}),
        on="circuitId", how="left", validate="many_to_one",
    )
    _require_rows(out, n, "join to circuits")
    if out["circuit"].isna().any():
        raise AssertionError("some races have a circuitId that is not in circuits")

    out = out.merge(
        qualifying[KEY + ["position"]].rename(columns={"position": "quali_position"}),
        on=KEY, how="left", validate="one_to_one",
    )
    _require_rows(out, n, "left join to qualifying")

    out["podium"] = out["positionOrder"].le(PODIUM_MAX_POSITION).fillna(False).astype(bool)
    out["front_row_grid"] = out["grid"].isin(FRONT_ROW_POSITIONS).fillna(False).astype(bool)
    out["front_row_quali"] = out["quali_position"].isin(FRONT_ROW_POSITIONS).fillna(False).astype(bool)
    return out.drop(columns=["positionOrder"])


def races_over_front_row_limit(df: pd.DataFrame, flag: str, limit: int = 2) -> pd.DataFrame:
    """Races that have more than `limit` rows with `flag` set (a front row holds two cars)."""
    counts = df[df[flag]].groupby("raceId").size().rename("n_front_row")
    bad = counts[counts > limit].reset_index()
    return bad.merge(df[["raceId", "year"]].drop_duplicates("raceId"), on="raceId", how="left")


def assign_era(year: pd.Series, split_year: int = config.DE_ERA_SPLIT_YEAR) -> pd.Series:
    return pd.Series(np.where(year < split_year, ERA_OLD, ERA_NEW), index=year.index)


def circuit_conversion(
    df: pd.DataFrame,
    flag: str,
    window: tuple[int, int] | None = None,
    min_starts: int = config.MIN_FRONT_ROW_STARTS,
    split_year: int = config.DE_ERA_SPLIT_YEAR,
) -> pd.DataFrame:
    """Podium conversion of front-row starts per circuit and era.

    One row per (circuit, era) with n_front_row, n_podium, rate, a Wilson 95% interval
    and `meets_min` (n_front_row >= min_starts).
    """
    sub = df if window is None else df[df["year"].between(*window)]
    sub = sub[sub[flag]].assign(era=lambda d: assign_era(d["year"], split_year))
    out = (
        sub.groupby(["circuitId", "circuit", "era"])["podium"]
        .agg(n_front_row="size", n_podium="sum")
        .reset_index()
    )
    out["n_podium"] = out["n_podium"].astype(int)
    out["rate"] = out["n_podium"] / out["n_front_row"]
    out["ci_low"], out["ci_high"] = wilson_interval(out["n_podium"], out["n_front_row"])
    out["meets_min"] = out["n_front_row"] >= min_starts
    return out


def conversion_tables(df: pd.DataFrame, min_starts: int = config.MIN_FRONT_ROW_STARTS) -> pd.DataFrame:
    """Circuit x era x definition tables for the primary and secondary windows, stacked."""
    parts = []
    for window_label, definition, flag, window in SPECS:
        part = circuit_conversion(df, flag, window=window, min_starts=min_starts)
        parts.append(part.assign(window=window_label, definition=definition))
    cols = ["window", "definition", "era", "circuitId", "circuit",
            "n_front_row", "n_podium", "rate", "ci_low", "ci_high", "meets_min"]
    return pd.concat(parts, ignore_index=True)[cols]


def threshold_report(conv: pd.DataFrame, thresholds: tuple[int, ...]) -> pd.DataFrame:
    """How many circuits (and front-row starts) each minimum-starts threshold drops."""
    rows = []
    for (window, definition, era), g in conv.groupby(["window", "definition", "era"], sort=False):
        for t in thresholds:
            kept = g["n_front_row"] >= t
            rows.append({
                "window": window, "definition": definition, "era": era, "min_starts": t,
                "circuits_with_starts": len(g),
                "circuits_kept": int(kept.sum()),
                "circuits_dropped": int((~kept).sum()),
                "starts_in_dropped": int(g.loc[~kept, "n_front_row"].sum()),
                "share_of_starts_dropped": g.loc[~kept, "n_front_row"].sum() / g["n_front_row"].sum(),
            })
    return pd.DataFrame(rows)


def two_proportion_test(s_old: int, n_old: int, s_new: int, n_new: int, z: float = Z_95) -> dict:
    """Compare two proportions: difference (new - old), Newcombe 95% interval, pooled two-sided z-test.

    The interval is Newcombe's hybrid score interval (built from the two Wilson intervals); the
    p-value is the usual pooled two-proportion z-test. Both treat every start as an independent
    trial. Starts from the same race (pole and P2) and from the same circuit are correlated, so
    the interval is somewhat too narrow and the p-value somewhat too small.
    """
    p_old, p_new = s_old / n_old, s_new / n_new
    l_old, u_old = wilson_interval(s_old, n_old, z)
    l_new, u_new = wilson_interval(s_new, n_new, z)
    diff = p_new - p_old
    ci_low = diff - sqrt((p_new - l_new) ** 2 + (u_old - p_old) ** 2)
    ci_high = diff + sqrt((u_new - p_new) ** 2 + (p_old - l_old) ** 2)

    pooled = (s_old + s_new) / (n_old + n_new)
    se = sqrt(pooled * (1 - pooled) * (1 / n_old + 1 / n_new))
    if se == 0:  # both proportions are 0 or both are 1: no difference to test
        z_stat, p_value = 0.0, 1.0
    else:
        z_stat = diff / se
        p_value = erfc(abs(z_stat) / sqrt(2))
    return {"diff": diff, "ci_low": ci_low, "ci_high": ci_high, "z": z_stat, "p_value": p_value}


def overall_era_difference(df: pd.DataFrame, window: tuple[int, int] = config.DE_WINDOW) -> pd.DataFrame:
    """Overall front-row conversion in the old vs new era, per definition, with the difference and test."""
    sub = df[df["year"].between(*window)].assign(era=lambda d: assign_era(d["year"]))
    rows = []
    for definition, flag in [(DEF_GRID, "front_row_grid"), (DEF_QUALI, "front_row_quali")]:
        starts = sub[sub[flag]]
        n = starts.groupby("era").size()
        k = starts.groupby("era")["podium"].sum()
        test = two_proportion_test(int(k[ERA_OLD]), int(n[ERA_OLD]), int(k[ERA_NEW]), int(n[ERA_NEW]))
        rows.append({
            "definition": definition,
            "starts_old": int(n[ERA_OLD]), "podiums_old": int(k[ERA_OLD]), "rate_old": k[ERA_OLD] / n[ERA_OLD],
            "starts_new": int(n[ERA_NEW]), "podiums_new": int(k[ERA_NEW]), "rate_new": k[ERA_NEW] / n[ERA_NEW],
            **test,
        })
    return pd.DataFrame(rows)


def spearman_interval(rho: float, n: int, z: float = Z_95) -> tuple[float, float]:
    """Approximate 95% interval for a Spearman correlation (Fisher z, Bonett-Wright standard error).

    Returns (nan, nan) when n <= 3 or rho is missing. It is an approximation: use it to see
    how uncertain a correlation from a small number of circuits is, not as an exact test.
    """
    if n <= 3 or rho is None or np.isnan(rho):
        return (np.nan, np.nan)
    se = np.sqrt((1 + rho**2 / 2) / (n - 3))
    with np.errstate(divide="ignore"):
        centre = np.arctanh(rho)
    return float(np.tanh(centre - z * se)), float(np.tanh(centre + z * se))


def ci_separated_pairs(ci_low, ci_high) -> tuple[int, int]:
    """(number of circuit pairs, number whose intervals do not overlap) for one era and definition."""
    lo = np.asarray(ci_low, dtype=float)
    hi = np.asarray(ci_high, dtype=float)
    i, j = np.triu_indices(len(lo), k=1)
    separated = (lo[i] > hi[j]) | (lo[j] > hi[i])
    return int(len(i)), int(separated.sum())


def era_rank_correlation(conv: pd.DataFrame, min_starts: int = config.MIN_FRONT_ROW_STARTS) -> pd.DataFrame:
    """Spearman correlation of circuit rates between eras, for circuits present in both eras.

    Reported twice: for every circuit present in both eras, and only for circuits that
    meet `min_starts` in both eras.
    """
    rows = []
    for (window, definition), g in conv.groupby(["window", "definition"], sort=False):
        rate = g.pivot(index="circuitId", columns="era", values="rate").reindex(columns=list(ERAS))
        n = g.pivot(index="circuitId", columns="era", values="n_front_row").reindex(columns=list(ERAS))
        both = rate.dropna()
        enough = both[(n.loc[both.index] >= min_starts).all(axis=1)]
        rho_both = both[ERA_OLD].corr(both[ERA_NEW], method="spearman") if len(both) > 2 else np.nan
        rho_min = enough[ERA_OLD].corr(enough[ERA_NEW], method="spearman") if len(enough) > 2 else np.nan
        lo_min, hi_min = spearman_interval(rho_min, len(enough))
        rows.append({
            "window": window, "definition": definition,
            "n_circuits_both_eras": len(both),
            "spearman_both_eras": rho_both,
            "n_circuits_min_both": len(enough),
            "spearman_min_both": rho_min,
            "spearman_min_both_ci_low": lo_min,
            "spearman_min_both_ci_high": hi_min,
        })
    return pd.DataFrame(rows)


def quali_coverage_by_season(df: pd.DataFrame, window: tuple[int, int] = config.DE_WINDOW) -> pd.DataFrame:
    """Share of cleaned rows per season that have a (non-null) qualifying position."""
    sub = df[df["year"].between(*window)]
    g = sub.groupby("year")["quali_position"]
    out = pd.DataFrame({"n_rows": g.size(), "n_with_quali": g.count()})
    out["coverage"] = out["n_with_quali"] / out["n_rows"]
    return out


def grid_vs_quali_cells(df: pd.DataFrame, window: tuple[int, int] = config.DE_WINDOW) -> pd.DataFrame:
    """Cross-tab of qualifying front row vs grid front row, with podium conversion per cell.

    Only rows that have a qualifying position are included. Cells where the two disagree
    show the effect of grid penalties, pit-lane starts and promotions.
    """
    sub = df[df["year"].between(*window) & df["quali_position"].notna()]
    q, g = sub["front_row_quali"], sub["front_row_grid"]
    cell = np.select(
        [q & g, q & ~g & sub["grid"].eq(0), q & ~g, ~q & g],
        ["front row on both", "quali front row, grid = 0 (pit lane / no grid slot)",
         "quali front row, grid 3+ (lost it on the grid)", "grid front row only (moved up)"],
        default="neither",
    )
    out = sub.assign(cell=cell).groupby("cell")["podium"].agg(n="size", n_podium="sum")
    out["n_podium"] = out["n_podium"].astype(int)
    out["rate"] = out["n_podium"] / out["n"]
    out["ci_low"], out["ci_high"] = wilson_interval(out["n_podium"], out["n"])
    return out.reset_index()
