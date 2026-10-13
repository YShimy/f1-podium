"""Programmatic leakage checks for the modeling table (used by tests/test_features.py and the notebook).

Each check returns plain data (a DataFrame or dict); the callers assert on it.

1. bruteforce_mismatches: recompute every lagged feature for sampled rows with naive code that
   only looks at rows strictly earlier by (year, round), independent of features.py.
2. perturbation_check: change race R's results, rebuild, compare.
3. standings_check: points_before vs the standings table of the previous round.
4. forbidden_columns: forbidden current-race columns present in X.
5. prefix_invariance_check: rows of race R built on data up to R only vs on all data.
6. split_check: splits disjoint, complete and ordered.
"""

import numpy as np
import pandas as pd

from f1podium import config
from f1podium import de_questions as de
from f1podium import features as F
from f1podium.eda import PODIUM_MAX_POSITION

LAGGED_FEATURES = [f for f in F.FEATURES if f.startswith(("drv_", "con_"))]


def _order(year, rnd):
    return np.asarray(year, dtype=np.int64) * 1000 + np.asarray(rnd, dtype=np.int64)


def bruteforce_features(tables: dict, results_clean: pd.DataFrame, rows: pd.DataFrame,
                        status_map: pd.DataFrame | None = None) -> pd.DataFrame:
    """Naive recomputation of LAGGED_FEATURES for `rows` (columns raceId, driverId, constructorId, year, round)."""
    status_map = de.load_status_categories() if status_map is None else status_map
    races = tables["races"][["raceId", "year", "round"]]
    res = results_clean.merge(races, on="raceId", how="left")
    res["ord"] = _order(res["year"], res["round"])
    res["podium"] = (res["positionOrder"] <= PODIUM_MAX_POSITION).astype(float)
    cat = tables["status"].merge(status_map, on="status", how="left").set_index("statusId")["category"]
    res["mech"] = (res["statusId"].map(cat) == de.MECHANICAL).astype(float)
    spr = tables["sprint_results"].merge(races, on="raceId", how="left")

    def points_before(entity, ident, year, rnd):
        r = res[(res[entity] == ident) & (res["year"] == year) & (res["round"] < rnd)]["points"].astype(float).sum()
        s = spr[(spr[entity] == ident) & (spr["year"] == year) & (spr["round"] < rnd)]["points"].astype(float).sum()
        return r + s

    def rank_in_field(entity, ident, race_id, year, rnd):
        if rnd == 1:
            return np.nan
        field = res.loc[res["raceId"] == race_id, entity].unique()
        pts = {e: points_before(entity, e, year, rnd) for e in field}
        return 1 + sum(v > pts[ident] for v in pts.values())

    out = []
    for r in rows.itertuples(index=False):
        o = _order(r.year, r.round)
        earlier = res[res["ord"] < o]
        rec = {"raceId": r.raceId, "driverId": r.driverId}
        rec["drv_points_before"] = points_before("driverId", r.driverId, r.year, r.round)
        rec["con_points_before"] = points_before("constructorId", r.constructorId, r.year, r.round)
        rec["drv_rank_before"] = rank_in_field("driverId", r.driverId, r.raceId, r.year, r.round)
        rec["con_rank_before"] = rank_in_field("constructorId", r.constructorId, r.raceId, r.year, r.round)

        d = earlier[earlier["driverId"] == r.driverId].sort_values("ord")
        rec["drv_races_before"] = len(d)
        c = earlier[earlier["constructorId"] == r.constructorId]
        c_races = c.drop_duplicates("raceId").sort_values("ord")["raceId"].tolist()
        rec["con_races_before"] = len(c_races)
        for k in config.ROLLING_WINDOWS:
            last = d.tail(k)
            rec[f"drv_podium_rate_last{k}"] = last["podium"].mean() if len(last) else np.nan
            rec[f"drv_mean_pos_last{k}"] = last["positionOrder"].astype(float).mean() if len(last) else np.nan
            cl = c[c["raceId"].isin(c_races[-k:])]
            rec[f"con_podium_rate_last{k}"] = cl["podium"].mean() if len(cl) else np.nan
            rec[f"con_mean_pos_last{k}"] = cl["positionOrder"].astype(float).mean() if len(cl) else np.nan
        cm = c[c["raceId"].isin(c_races[-config.MECH_WINDOW:])]
        rec[F.MECH_FEATURE] = cm["mech"].mean() if len(cm) else np.nan
        out.append(rec)
    return pd.DataFrame(out)


def bruteforce_mismatches(tables, results_clean, meta, X, n: int = 200, seed: int = config.RANDOM_SEED,
                          rows_idx=None) -> tuple[pd.DataFrame, int]:
    """(mismatching cells, number of rows checked) for a random sample of rows."""
    idx = meta.sample(n=min(n, len(meta)), random_state=seed).index if rows_idx is None else rows_idx
    naive = bruteforce_features(tables, results_clean, meta.loc[idx]).set_index(["raceId", "driverId"])
    built = pd.concat([meta.loc[idx, ["raceId", "driverId"]], X.loc[idx, LAGGED_FEATURES]], axis=1) \
        .set_index(["raceId", "driverId"])
    naive = naive.loc[built.index, LAGGED_FEATURES].astype(float)
    a, b = built.to_numpy(float), naive.to_numpy(float)
    same = np.isclose(a, b, atol=1e-9, rtol=0) | (np.isnan(a) & np.isnan(b))
    bad = np.argwhere(~same)
    mism = pd.DataFrame([{"raceId": built.index[i][0], "driverId": built.index[i][1],
                          "feature": LAGGED_FEATURES[j], "built": a[i, j], "naive": b[i, j]} for i, j in bad])
    return mism, len(idx)


def perturbation_check(tables, results_clean, race_id: int) -> dict:
    """Reverse race R's positionOrder, points and statusId among its rows, rebuild, compare with the original."""
    meta0, X0, y0 = F.build_modeling_table(tables, results_clean)
    pert = results_clean.copy()
    in_r = pert["raceId"] == race_id
    for col in ("positionOrder", "points", "statusId"):
        pert.loc[in_r, col] = pert.loc[in_r, col].to_numpy()[::-1]
    meta1, X1, y1 = F.build_modeling_table(tables, pert)
    assert meta0.equals(meta1)
    o = _order(meta0["year"], meta0["round"])
    r = meta0.loc[meta0["raceId"] == race_id]
    o_r = _order(r["year"].iloc[0], r["round"].iloc[0])
    differs = ~((X0 == X1) | (X0.isna() & X1.isna())).all(axis=1)
    return {
        "race": race_id,
        "rows_in_race": int(in_r.sum()),
        "changed_rows_up_to_and_incl_R": int(differs[o <= o_r].sum()),
        "changed_rows_after_R": int(differs[o > o_r].sum()),
        "changed_targets_in_R": int((y0 != y1)[meta0["raceId"] == race_id].sum()),
        "changed_targets_outside_R": int((y0 != y1)[meta0["raceId"] != race_id].sum()),
    }


def standings_check(tables, meta, X) -> pd.DataFrame:
    """Mismatches of points_before vs the previous round's standings, by decade (drivers and constructors).

    An entity missing from the previous round's standings is treated as 0 points (it had not raced yet).
    """
    races = tables["races"][["raceId", "year", "round"]]
    out = []
    for who, entity, table, feat in (("drivers", "driverId", "driver_standings", "drv_points_before"),
                                     ("constructors", "constructorId", "constructor_standings", "con_points_before")):
        prev = tables[table].merge(races, on="raceId", how="left")[[entity, "year", "round", "points"]]
        prev = prev.assign(round=prev["round"] + 1).rename(columns={"points": "standing"})
        rows = pd.concat([meta, X[[feat]]], axis=1)
        rows = rows[rows["round"] > 1].drop_duplicates(["raceId", entity])
        m = rows.merge(prev, on=[entity, "year", "round"], how="left", validate="many_to_one")
        m["standing"] = m["standing"].astype(float).fillna(0.0)
        m["mismatch"] = ~np.isclose(m[feat], m["standing"], atol=1e-6)
        g = m.groupby(m["year"] // 10 * 10).agg(rows=("mismatch", "size"), mismatches=("mismatch", "sum"))
        out.append(g.assign(who=who).reset_index().rename(columns={"year": "decade"}))
    return pd.concat(out, ignore_index=True)[["who", "decade", "rows", "mismatches"]]


def forbidden_columns(X: pd.DataFrame, y: pd.Series) -> list[str]:
    bad = set(X.columns) & set(config.FORBIDDEN_CURRENT_RACE_COLUMNS)
    if y.name in X.columns:
        bad.add(y.name)
    return sorted(bad)


def prefix_invariance_check(tables, results_clean, race_ids, full=None) -> pd.DataFrame:
    """For each race R: build on data with (year, round) <= R only; compare R's X rows with the full build."""
    meta, X, _ = F.build_modeling_table(tables, results_clean) if full is None else full
    races = tables["races"]
    o_all = _order(races["year"], races["round"])
    rows = []
    for race_id in race_ids:
        o_r = _order(*races.loc[races["raceId"] == race_id, ["year", "round"]].iloc[0])
        allowed = set(races.loc[o_all <= o_r, "raceId"])
        sub_tables = dict(tables)
        for name in ("qualifying", "sprint_results"):
            sub_tables[name] = tables[name][tables[name]["raceId"].isin(allowed)]
        sub_rc = results_clean[results_clean["raceId"].isin(allowed)]
        meta_p, X_p, _ = F.build_modeling_table(sub_tables, sub_rc)
        a = pd.concat([meta, X], axis=1).query("raceId == @race_id").set_index("driverId")[F.FEATURES]
        b = pd.concat([meta_p, X_p], axis=1).query("raceId == @race_id").set_index("driverId")[F.FEATURES]
        same = a.index.equals(b.index) and ((a == b) | (a.isna() & b.isna())).all().all()
        rows.append({"raceId": race_id, "rows": len(a), "rows_in_prefix_table": len(meta_p), "identical": bool(same)})
    return pd.DataFrame(rows)


def split_check(meta: pd.DataFrame) -> dict:
    years = {s: meta.loc[meta["split"] == s, "year"] for s in F.SPLITS}
    keys = {s: set(map(tuple, meta.loc[meta["split"] == s, ["raceId", "driverId"]].to_numpy())) for s in F.SPLITS}
    return {
        "ranges": {s: (int(v.min()), int(v.max()), len(v)) for s, v in years.items()},
        "disjoint": not (keys[F.SPLIT_TRAIN] & keys[F.SPLIT_VAL] or keys[F.SPLIT_VAL] & keys[F.SPLIT_TEST]
                         or keys[F.SPLIT_TRAIN] & keys[F.SPLIT_TEST]),
        "races_disjoint": meta.groupby("raceId")["split"].nunique().max() == 1,
        "complete": sum(len(v) for v in years.values()) == len(meta) and meta["split"].isin(F.SPLITS).all(),
        "ordered": years[F.SPLIT_TRAIN].max() < years[F.SPLIT_VAL].min()
                   and years[F.SPLIT_VAL].max() < years[F.SPLIT_TEST].min(),
    }
