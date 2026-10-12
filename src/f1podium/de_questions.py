"""Data-engineering questions: descriptive analyses of race outcomes.

These use race results (e.g. who finished on the podium) to *describe* the sport.
They are not model features, so the prediction cutoff (after qualifying, before the
race) does not apply to them.

Q1: which circuits convert a front-row start into a podium most often, and does the
ranking change between eras? Qualifying position and grid position are kept apart.

Q2: do drivers podium more often at a circuit in their home country than elsewhere,
controlling for starting position?

Q3: which constructors have the highest mechanical-retirement rate in 2014–2021 vs
2022–2024? `status` is only known after the race: it is used here to describe, and
must never be a model feature.
"""

from math import erfc, sqrt
from pathlib import Path

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


# ---------------------------------------------------------------------------
# Q2: do drivers podium more often in their home country, controlling for grid?
# ---------------------------------------------------------------------------

NO_CIRCUIT = "no circuit"
NATIONALITY_MAP_PATH = Path(__file__).resolve().parent / "reference" / "nationality_to_country.csv"
MAP_COLUMNS = ["nationality", "country", "note"]


def load_nationality_map(path: Path = NATIONALITY_MAP_PATH) -> pd.DataFrame:
    """Read the nationality -> circuits.country mapping (one row per pair; a nationality may have several).

    Text is stripped. Raises on wrong columns, empty values, duplicate pairs, or a nationality
    mapped both to a country and to "no circuit".
    """
    m = pd.read_csv(path, dtype=str, keep_default_na=False)
    if list(m.columns) != MAP_COLUMNS:
        raise ValueError(f"mapping columns must be {MAP_COLUMNS}, got {list(m.columns)}")
    for col in ("nationality", "country"):
        m[col] = m[col].str.strip()
    if (m[["nationality", "country"]] == "").any().any():
        raise ValueError("mapping has empty nationality or country values")
    if m.duplicated(["nationality", "country"]).any():
        raise ValueError("mapping has duplicate (nationality, country) pairs")
    no_circuit = m["country"] == NO_CIRCUIT
    if m.loc[no_circuit, "nationality"].isin(m.loc[~no_circuit, "nationality"]).any():
        raise ValueError("a nationality is mapped both to a country and to 'no circuit'")
    return m


def check_nationality_map(mapping: pd.DataFrame, drivers: pd.DataFrame, circuits: pd.DataFrame) -> dict:
    """Completeness of the mapping against the data.

    unmapped_nationalities: driver nationalities (stripped) with no mapping row.
    unknown_countries: mapped countries that are neither a circuits.country value nor "no circuit".
    """
    nationalities = set(drivers["nationality"].dropna().str.strip())
    valid_countries = set(circuits["country"].dropna()) | {NO_CIRCUIT}
    return {
        "unmapped_nationalities": sorted(nationalities - set(mapping["nationality"])),
        "unknown_countries": sorted(set(mapping["country"]) - valid_countries),
        "drivers_without_nationality": int(drivers["nationality"].isna().sum()),
    }


def grid_bucket_labels() -> list[str]:
    return [config.GRID_BUCKET_PIT_LANE] + [label for label, _, _ in config.GRID_BUCKETS]


def assign_grid_bucket(grid: pd.Series) -> pd.Series:
    """Ordered grid bucket per row: 0 -> pit lane, then config.GRID_BUCKETS. Missing grid stays missing."""
    g = pd.to_numeric(grid)
    present = g.dropna()
    if (present < 0).any() or (present != present.round()).any():
        raise ValueError("grid must be a non-negative whole number")
    out = pd.Series(pd.NA, index=grid.index, dtype="object")
    out[g.eq(0).fillna(False).to_numpy(bool)] = config.GRID_BUCKET_PIT_LANE
    for label, first, last in config.GRID_BUCKETS:
        in_bucket = g.ge(first) if last is None else (g.ge(first) & g.le(last))
        out[in_bucket.fillna(False).to_numpy(bool)] = label
    if out[g.notna().to_numpy(bool)].isna().any():
        raise ValueError("config.GRID_BUCKETS does not cover every grid value")
    return pd.Series(pd.Categorical(out, categories=grid_bucket_labels(), ordered=True), index=grid.index)


def build_home_table(
    results_clean: pd.DataFrame,
    races: pd.DataFrame,
    circuits: pd.DataFrame,
    drivers: pd.DataFrame,
    mapping: pd.DataFrame,
) -> pd.DataFrame:
    """One row per (raceId, driverId) with circuit country, driver nationality and a `home` flag.

    Joins: cleaned results -> races (year, circuitId) -> circuits (country) -> drivers (nationality,
    stripped). `home` is True when (nationality, circuit country) is a pair in the mapping; this is a
    lookup on unique pairs, so a nationality mapped to several countries never multiplies rows.
    Raises if a key is not unique or any join changes the row count.
    """
    _require_unique(results_clean, KEY, "results_clean")
    _require_unique(races, ["raceId"], "races")
    _require_unique(circuits, ["circuitId"], "circuits")
    _require_unique(drivers, ["driverId"], "drivers")
    _require_unique(mapping, ["nationality", "country"], "nationality mapping")

    n = len(results_clean)
    out = results_clean[KEY + ["grid", "positionOrder"]].copy()

    out = out.merge(races[["raceId", "year", "circuitId"]], on="raceId", how="left", validate="many_to_one")
    _require_rows(out, n, "join to races")
    out = out.merge(
        circuits[["circuitId", "country"]].rename(columns={"country": "circuit_country"}),
        on="circuitId", how="left", validate="many_to_one",
    )
    _require_rows(out, n, "join to circuits")
    out = out.merge(drivers[["driverId", "nationality"]], on="driverId", how="left", validate="many_to_one")
    _require_rows(out, n, "join to drivers")
    if out[["year", "circuit_country"]].isna().any().any():
        raise AssertionError("missing year or circuit country after the joins")
    out["nationality"] = out["nationality"].str.strip()

    home_pairs = (
        mapping.loc[mapping["country"] != NO_CIRCUIT, ["nationality", "country"]]
        .rename(columns={"country": "circuit_country"})
        .assign(_home=True)
    )
    out = out.merge(home_pairs, on=["nationality", "circuit_country"], how="left", validate="many_to_one")
    _require_rows(out, n, "home lookup")

    out["home"] = out["_home"].notna()
    out["mapped"] = out["nationality"].isin(mapping["nationality"])
    out["has_home_circuit"] = out["nationality"].isin(home_pairs["nationality"])
    out["decade"] = (out["year"] // 10 * 10).astype(int)
    out["grid_bucket"] = assign_grid_bucket(out["grid"])
    out["podium"] = out["positionOrder"].le(PODIUM_MAX_POSITION).fillna(False).astype(bool)
    return out.drop(columns=["_home", "positionOrder"])


def bucket_rates(df: pd.DataFrame) -> pd.DataFrame:
    """Podium rate per grid bucket x home/away, with n, podiums and a Wilson 95% interval."""
    out = (
        df.dropna(subset=["grid_bucket"])
        .assign(side=lambda d: np.where(d["home"], "home", "away"))
        .groupby(["grid_bucket", "side"], observed=True)["podium"]
        .agg(n="size", podiums="sum")
        .reset_index()
    )
    out["podiums"] = out["podiums"].astype(int)
    out["rate"] = out["podiums"] / out["n"]
    out["ci_low"], out["ci_high"] = wilson_interval(out["podiums"], out["n"])
    return out


def _informative_strata(df: pd.DataFrame, strata: str) -> tuple[pd.DataFrame, list, list]:
    """Keep strata with home and away rows and both outcomes; the others carry no information on the OR."""
    used, dropped = [], []
    for value, g in df.dropna(subset=[strata]).groupby(strata, observed=True):
        has_both_sides = g["home"].any() and (~g["home"]).any()
        has_both_outcomes = g["podium"].any() and (~g["podium"]).any()
        (used if has_both_sides and has_both_outcomes else dropped).append(value)
    return df[df[strata].isin(used)], used, dropped


def cmh_home_or(df: pd.DataFrame, strata: str = "grid_bucket") -> dict:
    """Mantel-Haenszel odds ratio of a podium at home vs away, pooled over grid buckets.

    Also returns the 95% CI, the p-value for OR = 1 (CMH test) and the Breslow-Day p-value for equal
    odds ratios across strata. Strata with no home or no away rows, or with no podiums (or only
    podiums), carry no information on the OR and are dropped; they are listed in the result.
    Rows are treated as independent (no clustering by driver).
    """
    from statsmodels.stats.contingency_tables import StratifiedTable

    sub, used, dropped = _informative_strata(df, strata)
    tables = []
    for value in used:
        g = sub[sub[strata] == value]
        h, p = g["home"], g["podium"]
        tables.append(np.array([[(h & p).sum(), (h & ~p).sum()], [(~h & p).sum(), (~h & ~p).sum()]], dtype=float))
    st = StratifiedTable(tables)
    low, high = st.oddsratio_pooled_confint(alpha=0.05)
    return {
        "odds_ratio": float(st.oddsratio_pooled), "ci_low": float(low), "ci_high": float(high),
        "p_value": float(st.test_null_odds().pvalue),
        "homogeneity_p": float(st.test_equal_odds().pvalue) if len(tables) > 1 else np.nan,
        "n_rows": len(sub), "n_home": int(sub["home"].sum()),
        "strata_used": [str(v) for v in used], "strata_dropped": [str(v) for v in dropped],
    }


def logit_home_or(df: pd.DataFrame) -> dict:
    """Odds ratio for `home` from podium ~ home + C(grid_bucket) + C(decade), SEs clustered by driver.

    Grid buckets without both outcomes (the pit-lane bucket has no podiums) cause perfect
    separation and carry no information on `home`, so they are dropped and listed.
    """
    import statsmodels.formula.api as smf

    sub, _, dropped = _informative_strata(df, "grid_bucket")
    data = pd.DataFrame({
        "podium": sub["podium"].astype(int).to_numpy(),
        "home": sub["home"].astype(int).to_numpy(),
        "grid_bucket": sub["grid_bucket"].astype(str).to_numpy(),
        "decade": sub["decade"].astype(str).to_numpy(),
        "driverId": sub["driverId"].astype(int).to_numpy(),
    })
    model = smf.logit('podium ~ home + C(grid_bucket, Treatment(reference="1")) + C(decade)', data=data)
    res = model.fit(disp=0, cov_type="cluster", cov_kwds={"groups": data["driverId"]})
    low, high = res.conf_int().loc["home"]
    return {
        "odds_ratio": float(np.exp(res.params["home"])), "ci_low": float(np.exp(low)), "ci_high": float(np.exp(high)),
        "p_value": float(res.pvalues["home"]), "n_rows": int(res.nobs), "n_home": int(data["home"].sum()),
        "n_drivers": int(data["driverId"].nunique()), "converged": bool(res.mle_retvals["converged"]),
        "strata_dropped": [str(v) for v in dropped],
    }


def sensitivity_subsets(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """(a) drivers with >= Q2_MIN_HOME_RACES home and >= Q2_MIN_AWAY_RACES away races; (b) recent seasons."""
    per_driver = df.groupby("driverId")["home"].agg(n_home="sum", n_rows="size")
    per_driver["n_away"] = per_driver["n_rows"] - per_driver["n_home"]
    eligible = per_driver.index[
        (per_driver["n_home"] >= config.Q2_MIN_HOME_RACES) & (per_driver["n_away"] >= config.Q2_MIN_AWAY_RACES)
    ]
    return {
        f"(a) drivers with >= {config.Q2_MIN_HOME_RACES} home and >= {config.Q2_MIN_AWAY_RACES} away races":
            df[df["driverId"].isin(eligible)],
        f"(b) {config.Q2_RECENT_START_YEAR}+ only": df[df["year"] >= config.Q2_RECENT_START_YEAR],
    }


def q2_effects(df: pd.DataFrame) -> pd.DataFrame:
    """One row per analysis: main CMH, main clustered logistic, and CMH for each sensitivity subset."""
    rows = [{"analysis": "main (all years)", "method": "CMH, stratified by grid bucket", **cmh_home_or(df)}]
    rows.append({"analysis": "main (all years)", "method": "logistic + grid bucket + decade, clustered by driver",
                 **logit_home_or(df)})
    for name, subset in sensitivity_subsets(df).items():
        rows.append({"analysis": name, "method": "CMH, stratified by grid bucket", **cmh_home_or(subset)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Q3: mechanical-retirement rate per constructor, 2014–2021 vs 2022–2024
# ---------------------------------------------------------------------------

REFERENCE_DIR = Path(__file__).resolve().parent / "reference"
STATUS_CATEGORY_PATH = REFERENCE_DIR / "status_category.csv"
CONSTRUCTOR_LINEAGE_PATH = REFERENCE_DIR / "constructor_lineage.csv"
STATUS_CATEGORIES = ("finished", "mechanical", "accident_driver", "regulatory", "other")
STATUS_MAP_COLUMNS = ["status", "category", "ambiguous", "alt_category", "note"]
LINEAGE_COLUMNS = ["constructorId", "constructorRef", "lineage", "note"]
MECHANICAL = "mechanical"


def load_status_categories(path: Path = STATUS_CATEGORY_PATH) -> pd.DataFrame:
    """Read the status -> category mapping. `alt_category` (sensitivity) is set exactly on ambiguous rows.

    Raises on wrong columns, duplicate statuses, unknown categories, or an alt_category that is
    missing on an ambiguous row, present on a clear-cut row, or equal to the main category.
    In the returned frame, clear-cut rows get alt_category = category.
    """
    m = pd.read_csv(path, dtype=str, keep_default_na=False)
    if list(m.columns) != STATUS_MAP_COLUMNS:
        raise ValueError(f"status mapping columns must be {STATUS_MAP_COLUMNS}, got {list(m.columns)}")
    if (m["status"] == "").any():
        raise ValueError("status mapping has an empty status")
    if m["status"].duplicated().any():
        raise ValueError(f"duplicate statuses: {sorted(m.loc[m['status'].duplicated(), 'status'])}")
    if not m["ambiguous"].isin(["True", "False"]).all():
        raise ValueError("ambiguous must be True or False")
    m["ambiguous"] = m["ambiguous"] == "True"
    if not m["category"].isin(STATUS_CATEGORIES).all():
        raise ValueError(f"unknown category; allowed: {STATUS_CATEGORIES}")
    alt_ok = np.where(m["ambiguous"],
                      m["alt_category"].isin(STATUS_CATEGORIES) & (m["alt_category"] != m["category"]),
                      m["alt_category"] == "")
    if not alt_ok.all():
        raise ValueError(f"bad alt_category for: {sorted(m.loc[~alt_ok, 'status'])}")
    m["alt_category"] = m["alt_category"].where(m["ambiguous"], m["category"])
    return m


def check_status_categories(mapping: pd.DataFrame, status: pd.DataFrame) -> dict:
    """Coverage of the mapping against status.csv: statuses with no mapping, and mapped names not in the data."""
    names = set(status["status"].dropna())
    return {
        "unmapped_statuses": sorted(names - set(mapping["status"])),
        "unknown_statuses": sorted(set(mapping["status"]) - names),
    }


def load_constructor_lineage(path: Path = CONSTRUCTOR_LINEAGE_PATH) -> pd.DataFrame:
    """Read the constructorId -> lineage mapping (renamed or bought teams merged). Unique on constructorId."""
    m = pd.read_csv(path, dtype={"constructorId": "Int64"}, keep_default_na=False)
    if list(m.columns) != LINEAGE_COLUMNS:
        raise ValueError(f"lineage columns must be {LINEAGE_COLUMNS}, got {list(m.columns)}")
    _require_unique(m, ["constructorId"], "constructor lineage")
    if (m["lineage"].str.strip() == "").any():
        raise ValueError("lineage has empty values")
    return m


def assign_q3_era(year: pd.Series, eras=config.Q3_ERAS) -> pd.Series:
    """Era label per year from config.Q3_ERAS; years outside every era are missing."""
    out = pd.Series(pd.NA, index=year.index, dtype="object")
    for label, first, last in eras:
        out[year.between(first, last).fillna(False).to_numpy(bool)] = label
    return out


def build_retirement_table(
    results_clean: pd.DataFrame,
    races: pd.DataFrame,
    constructors: pd.DataFrame,
    status: pd.DataFrame,
    mapping: pd.DataFrame,
    eras=config.Q3_ERAS,
) -> pd.DataFrame:
    """One row per (raceId, driverId) of the cleaned results in the Q3 eras, with constructor and status category.

    Joins: cleaned results -> races (year) -> constructors (name) -> status (text) -> mapping (category).
    Every join is many-to-one and is checked for an unchanged row count; a null statusId, an unknown
    raceId/constructorId/statusId or an unmapped status raises. The era filter is applied last.

    Columns added: year, era, constructor, status, category, alt_category, ambiguous,
    mechanical (category == mechanical) and mechanical_alt (ambiguous statuses flipped).
    """
    _require_unique(results_clean, KEY, "results_clean")
    _require_unique(races, ["raceId"], "races")
    _require_unique(constructors, ["constructorId"], "constructors")
    _require_unique(status, ["statusId"], "status")
    _require_unique(mapping, ["status"], "status mapping")
    if results_clean["statusId"].isna().any():
        raise AssertionError("cleaned results have a null statusId")

    n = len(results_clean)
    out = results_clean[KEY + ["constructorId", "statusId"]].copy()
    out = out.merge(races[["raceId", "year"]], on="raceId", how="left", validate="many_to_one")
    _require_rows(out, n, "join to races")
    out = out.merge(constructors[["constructorId", "name"]].rename(columns={"name": "constructor"}),
                    on="constructorId", how="left", validate="many_to_one")
    _require_rows(out, n, "join to constructors")
    out = out.merge(status[["statusId", "status"]], on="statusId", how="left", validate="many_to_one")
    _require_rows(out, n, "join to status")
    out = out.merge(mapping[["status", "category", "alt_category", "ambiguous"]], on="status",
                    how="left", validate="many_to_one")
    _require_rows(out, n, "join to status mapping")
    for col, what in [("year", "a raceId not in races"), ("constructor", "a constructorId not in constructors"),
                      ("status", "a statusId not in status"), ("category", "a status not in the mapping")]:
        if out[col].isna().any():
            raise AssertionError(f"{int(out[col].isna().sum())} rows with {what}")

    out["era"] = assign_q3_era(out["year"], eras)
    out = out[out["era"].notna()].reset_index(drop=True)
    out["ambiguous"] = out["ambiguous"].astype(bool)
    out["mechanical"] = out["category"] == MECHANICAL
    out["mechanical_alt"] = out["alt_category"] == MECHANICAL
    return out


def add_lineage(df: pd.DataFrame, lineage: pd.DataFrame) -> pd.DataFrame:
    """Add a `lineage` column; raises if a constructorId in df is not in the lineage file."""
    out = df.merge(lineage[["constructorId", "lineage"]], on="constructorId", how="left", validate="many_to_one")
    _require_rows(out, len(df), "join to lineage")
    if out["lineage"].isna().any():
        raise AssertionError("constructorIds missing from the lineage file: "
                             f"{sorted(out.loc[out['lineage'].isna(), 'constructorId'].unique())}")
    return out


def mechanical_rates(df: pd.DataFrame, by: str = "constructorId", flag: str = "mechanical",
                     min_starts: int = config.Q3_MIN_STARTS) -> pd.DataFrame:
    """Per group x era: starts, mechanical retirements, rate, Wilson 95% interval and `meets_min`.

    `label` is the display name: the constructor's name (its latest in the era) or the lineage itself.
    """
    label_col = "constructor" if by == "constructorId" else by
    out = (
        df.sort_values("year", kind="stable")
        .groupby([by, "era"], observed=True)
        .agg(label=(label_col, "last"), starts=(flag, "size"), n_mech=(flag, "sum"))
        .reset_index()
    )
    out["n_mech"] = out["n_mech"].astype(int)
    out["rate"] = out["n_mech"] / out["starts"]
    out["ci_low"], out["ci_high"] = wilson_interval(out["n_mech"], out["starts"])
    out["meets_min"] = out["starts"] >= min_starts
    return out


def overall_mechanical_change(df: pd.DataFrame, flag: str = "mechanical", eras=config.Q3_ERAS) -> dict:
    """Overall mechanical-retirement rate in the first vs second era, with the two-proportion test."""
    (old, *_), (new, *_) = eras
    n = df.groupby("era").size()
    k = df.groupby("era")[flag].sum()
    test = two_proportion_test(int(k[old]), int(n[old]), int(k[new]), int(n[new]))
    return {"starts_old": int(n[old]), "mech_old": int(k[old]), "rate_old": k[old] / n[old],
            "starts_new": int(n[new]), "mech_new": int(k[new]), "rate_new": k[new] / n[new], **test}


def mechanical_rank_correlation(rates: pd.DataFrame, by: str = "constructorId", eras=config.Q3_ERAS) -> dict:
    """Spearman correlation of the group rates between eras, for groups meeting the minimum in both eras."""
    (old, *_), (new, *_) = eras
    kept = rates[rates["meets_min"]]
    wide = kept.pivot(index=by, columns="era", values="rate").reindex(columns=[old, new]).dropna()
    rho = wide[old].corr(wide[new], method="spearman") if len(wide) > 2 else np.nan
    low, high = spearman_interval(rho, len(wide))
    return {"n_both_eras": len(wide), "spearman": rho, "ci_low": low, "ci_high": high}
