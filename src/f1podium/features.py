"""Modeling table: one row per (raceId, driverId) of the cleaned results, with leakage-safe features.

Prediction cutoff: after qualifying, before the race starts. For the current race only the grid,
qualifying and static metadata are used. Everything else comes from EARLIER races, ordered by
(year, round) -- never by raceId, which is not chronological -- and shifted by one race before any
cumulative or rolling statistic. Nothing is imputed, scaled or encoded here: missing values stay
NaN and are handled later by pipelines fitted on the training data only.

build_modeling_table returns (meta, X, y) with the same index:
  meta: raceId, driverId, constructorId, constructor_key, year, round, split (never features)

Constructor features (con_*) are keyed by config.CONSTRUCTOR_KEY: the team lineage by default
(renamed or bought teams merged), or constructorId for an ablation.
  X:    the features listed in FEATURES
  y:    podium (positionOrder <= 3), kept out of X
"""

import numpy as np
import pandas as pd

from f1podium import config
from f1podium import de_questions as de
from f1podium.eda import PODIUM_MAX_POSITION

KEY = ["raceId", "driverId"]
META_COLUMNS = ["raceId", "driverId", "constructorId", "constructor_key", "year", "round", "split"]
CONSTRUCTOR_KEYS = ("lineage", "constructorId")
SPLIT_TRAIN, SPLIT_VAL, SPLIT_TEST = "train", "validation", "test"
SPLITS = (SPLIT_TRAIN, SPLIT_VAL, SPLIT_TEST)

# (feature, definition, source, availability at the cutoff, window)
_W = config.ROLLING_WINDOWS
FEATURE_DICTIONARY = [
    ("grid_filled", "starting grid position; grid = 0 (pit lane) -> number of starters in the race",
     "results.grid", "known before the start", "race R"),
    ("pit_lane_start", "1 if grid = 0", "results.grid", "known before the start", "race R"),
    ("quali_position", "qualifying position (NaN when there is no qualifying row)", "qualifying.position",
     "known after qualifying", "race R"),
    ("has_quali", "1 if quali_position is present", "qualifying", "known after qualifying", "race R"),
    ("is_round_1", "1 for the first round of the season", "races.round", "static", "race R"),
    ("is_home", "driver nationality maps to the circuit's country (Q2 mapping)",
     "drivers, circuits, reference/nationality_to_country.csv", "static", "race R"),
    ("drv_points_before", "driver's season points (race + sprint) from rounds before R; 0 at round 1",
     "results.points, sprint_results.points", "earlier rounds only", "season to date"),
    ("drv_rank_before", "rank of drv_points_before among race R's entrants (ties share the best rank); NaN at round 1",
     "derived", "earlier rounds only", "season to date"),
    ("con_points_before", "constructor (lineage) season points (race + sprint, all cars) from rounds before R; 0 at round 1",
     "results.points, sprint_results.points", "earlier rounds only", "season to date"),
    ("con_rank_before", "rank of con_points_before among race R's constructors (lineages); NaN at round 1",
     "derived", "earlier rounds only", "season to date"),
    *[(f"drv_podium_rate_last{k}", f"driver's podium share over their previous {k} starts (fewer if not available)",
       "results.positionOrder", "earlier races only", f"last {k} starts, across seasons") for k in _W],
    *[(f"drv_mean_pos_last{k}", f"driver's mean finishing order over their previous {k} starts",
       "results.positionOrder", "earlier races only", f"last {k} starts, across seasons") for k in _W],
    ("drv_races_before", "number of earlier starts in the driver's career", "results", "earlier races only", "career"),
    *[(f"con_podium_rate_last{k}", f"constructor's podiums / starts over its previous {k} races (all cars)",
       "results.positionOrder", "earlier races only", f"last {k} races, across seasons") for k in _W],
    *[(f"con_mean_pos_last{k}", f"constructor's mean finishing order over its previous {k} races (all cars)",
       "results.positionOrder", "earlier races only", f"last {k} races, across seasons") for k in _W],
    ("con_races_before", "number of earlier races for the constructor", "results", "earlier races only", "career"),
    (f"con_mech_rate_last{config.MECH_WINDOW}",
     f"constructor's mechanical retirements / starts over its previous {config.MECH_WINDOW} races",
     "results.statusId of earlier races, reference/status_category.csv", "earlier races only",
     f"last {config.MECH_WINDOW} races"),
]
FEATURES = [row[0] for row in FEATURE_DICTIONARY]
MECH_FEATURE = f"con_mech_rate_last{config.MECH_WINDOW}"


def feature_dictionary() -> pd.DataFrame:
    return pd.DataFrame(FEATURE_DICTIONARY,
                        columns=["feature", "definition", "source", "availability", "window"]).set_index("feature")


def assign_split(year: pd.Series) -> pd.Series:
    """train <= TRAIN_END_YEAR, validation in VAL_YEARS, test >= TEST_START_YEAR (config)."""
    first_val, last_val = config.VAL_YEARS
    out = np.select(
        [year <= config.TRAIN_END_YEAR, year.between(first_val, last_val), year >= config.TEST_START_YEAR],
        list(SPLITS), default="",
    )
    if (out == "").any():
        raise AssertionError("config split years leave a gap")
    return pd.Series(out, index=year.index)


def _require_rows(df: pd.DataFrame, expected: int, step: str) -> None:
    if len(df) != expected:
        raise AssertionError(f"row count changed after {step}: {expected} -> {len(df)}")


def chronological(results_clean: pd.DataFrame, races: pd.DataFrame) -> pd.DataFrame:
    """Results with year and round, sorted by (year, round, driverId) -- never by raceId."""
    if races.duplicated(["year", "round"]).any():
        raise AssertionError("(year, round) is not unique in races")
    if results_clean.duplicated(KEY).any():
        raise AssertionError("results_clean is not unique on (raceId, driverId)")
    out = results_clean.merge(races[["raceId", "year", "round", "circuitId"]], on="raceId",
                              how="left", validate="many_to_one")
    _require_rows(out, len(results_clean), "join to races")
    if out[["year", "round"]].isna().any().any():
        raise AssertionError("some results have a raceId that is not in races")
    return out.sort_values(["year", "round", "driverId"], kind="stable").reset_index(drop=True)


def season_points_before(events: pd.DataFrame, entity: str) -> pd.DataFrame:
    """Season-to-date points from EARLIER rounds per (entity, year, round).

    `events` has one or more rows per (entity, year, round) with a `pts` column (race and sprint
    points). They are summed per round, cumulated within (entity, year) in round order, and the
    current round is subtracted, so the result only contains rounds < R.
    """
    per_round = (events.groupby([entity, "year", "round"], as_index=False)["pts"].sum()
                 .sort_values([entity, "year", "round"], kind="stable"))
    cum = per_round.groupby([entity, "year"])["pts"].cumsum()
    per_round["points_before"] = cum - per_round["pts"]
    return per_round[[entity, "year", "round", "points_before"]]


def rolling_prior(seq: pd.DataFrame, entity: str, sums: list[str], windows) -> pd.DataFrame:
    """Rolling sums of `sums` over the previous k rows of each entity (seq is one row per entity-race).

    Sorted by (year, round) within the entity, shifted by one row (so the current race is never
    included), then summed over a window of k with min_periods=1. Also returns races_before.
    """
    seq = seq.sort_values([entity, "year", "round"], kind="stable")
    g = seq.groupby(entity, sort=False)
    out = seq[[entity, "raceId"]].copy()
    out["races_before"] = g.cumcount()
    for col in sums:
        prior = g[col].shift(1)
        for k in windows:
            out[f"{col}_sum{k}"] = (prior.groupby(seq[entity], sort=False)
                                    .rolling(k, min_periods=1).sum().reset_index(level=0, drop=True))
    return out


def _status_mechanical(status: pd.DataFrame, status_map: pd.DataFrame) -> pd.DataFrame:
    s = status[["statusId", "status"]].merge(status_map[["status", "category"]], on="status",
                                             how="left", validate="one_to_one")
    if s["category"].isna().any():
        raise AssertionError(f"statuses without a category: {sorted(s.loc[s['category'].isna(), 'status'])}")
    return s.assign(mech=(s["category"] == de.MECHANICAL).astype(int))[["statusId", "mech"]]


def build_modeling_table(
    tables: dict[str, pd.DataFrame],
    results_clean: pd.DataFrame,
    status_map: pd.DataFrame | None = None,
    nat_map: pd.DataFrame | None = None,
    constructor_key: str = config.CONSTRUCTOR_KEY,
    lineage: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Build (meta, X, y) from the cleaned results and the raw tables (see the module docstring).

    Every con_* feature is keyed by `constructor_key`: "lineage" (renamed or bought teams merged, from
    reference/constructor_lineage.csv by constructorId and year) or "constructorId" (ablation). The key
    is stored in meta["constructor_key"]; it is never a feature.
    """
    if constructor_key not in CONSTRUCTOR_KEYS:
        raise ValueError(f"constructor_key must be one of {CONSTRUCTOR_KEYS}")
    status_map = de.load_status_categories() if status_map is None else status_map
    nat_map = de.load_nationality_map() if nat_map is None else nat_map
    lineage = de.load_constructor_lineage() if lineage is None else lineage

    def con_key_for(frame: pd.DataFrame) -> pd.Series:
        if constructor_key == "constructorId":
            return frame["constructorId"].astype("int64").astype(str)
        return de.lineage_for(frame, lineage, tables["constructors"])
    races, qualifying, sprint = tables["races"], tables["qualifying"], tables["sprint_results"]

    base = chronological(results_clean, races)
    n = len(base)
    df = base[["raceId", "driverId", "constructorId", "year", "round", "grid", "positionOrder", "points",
               "statusId"]].copy()
    df["constructor_key"] = con_key_for(df)

    # --- current race: grid, qualifying, static --------------------------------------------
    n_starters = df.groupby("raceId")["driverId"].transform("size")
    df["pit_lane_start"] = df["grid"].eq(0).astype(int)
    df["grid_filled"] = df["grid"].where(df["grid"] != 0, n_starters).astype(float)
    q = qualifying[KEY + ["position"]].rename(columns={"position": "quali_position"})
    df = df.merge(q, on=KEY, how="left", validate="one_to_one")
    _require_rows(df, n, "left join to qualifying")
    df["quali_position"] = df["quali_position"].astype(float)
    df["has_quali"] = df["quali_position"].notna().astype(int)
    df["is_round_1"] = df["round"].eq(1).astype(int)
    home = de.build_home_table(results_clean, races, tables["circuits"], tables["drivers"], nat_map)[KEY + ["home"]]
    df = df.merge(home, on=KEY, how="left", validate="one_to_one")
    _require_rows(df, n, "join to home flag")
    df["is_home"] = df.pop("home").astype(int)

    # --- season points before round R (race + sprint of EARLIER rounds) ---------------------
    sprint_y = sprint.merge(races[["raceId", "year", "round"]], on="raceId", how="left", validate="many_to_one")
    sprint_y["constructor_key"] = con_key_for(sprint_y)
    for prefix, entity in (("drv", "driverId"), ("con", "constructor_key")):
        events = pd.concat([
            df[[entity, "year", "round", "points"]].rename(columns={"points": "pts"}),
            sprint_y[[entity, "year", "round", "points"]].rename(columns={"points": "pts"}),
        ], ignore_index=True)
        events["pts"] = events["pts"].astype(float)
        before = season_points_before(events, entity).rename(columns={"points_before": f"{prefix}_points_before"})
        df = df.merge(before, on=[entity, "year", "round"], how="left", validate="many_to_one")
        _require_rows(df, n, f"join {prefix} points_before")

    # Rank within race R's field (drivers; constructors ranked once per constructor key).
    df["drv_rank_before"] = df.groupby("raceId")["drv_points_before"].rank(ascending=False, method="min")
    con_field = df.drop_duplicates(["raceId", "constructor_key"])[["raceId", "constructor_key", "con_points_before"]]
    con_field = con_field.assign(con_rank_before=con_field.groupby("raceId")["con_points_before"]
                                 .rank(ascending=False, method="min"))
    df = df.merge(con_field[["raceId", "constructor_key", "con_rank_before"]], on=["raceId", "constructor_key"],
                  how="left", validate="many_to_one")
    _require_rows(df, n, "join con_rank_before")
    df.loc[df["round"] == 1, ["drv_rank_before", "con_rank_before"]] = np.nan

    # --- driver form: previous starts --------------------------------------------------------
    df["_podium"] = (df["positionOrder"] <= PODIUM_MAX_POSITION).astype(float)
    df["_pos"] = df["positionOrder"].astype(float)
    df["_one"] = 1.0
    drv = rolling_prior(df[["driverId", "raceId", "year", "round", "_podium", "_pos", "_one"]],
                        "driverId", ["_podium", "_pos", "_one"], _W)
    for k in _W:
        drv[f"drv_podium_rate_last{k}"] = drv[f"_podium_sum{k}"] / drv[f"_one_sum{k}"]
        drv[f"drv_mean_pos_last{k}"] = drv[f"_pos_sum{k}"] / drv[f"_one_sum{k}"]
    drv = drv.rename(columns={"races_before": "drv_races_before"})
    keep = ["driverId", "raceId", "drv_races_before"] + [c for c in drv.columns if c.startswith("drv_")
                                                         and c != "drv_races_before"]
    df = df.merge(drv[keep], on=KEY, how="left", validate="one_to_one")
    _require_rows(df, n, "join driver form")

    # --- constructor form and reliability ------------------------------------------------------
    # Teammate safety: rows are first aggregated to ONE row per (constructor key, race), and only then
    # shifted and rolled, so no car's result in race R (its own or a teammate's) enters race R's features.
    df = df.merge(_status_mechanical(tables["status"], status_map), on="statusId", how="left", validate="many_to_one")
    _require_rows(df, n, "join status category")
    if df["mech"].isna().any():
        raise AssertionError("some results have a statusId that is not in status")
    con_race = (df.groupby(["constructor_key", "raceId", "year", "round"], as_index=False)
                .agg(_podium=("_podium", "sum"), _pos=("_pos", "sum"), _one=("_one", "sum"), _mech=("mech", "sum")))
    con = rolling_prior(con_race, "constructor_key", ["_podium", "_pos", "_one"], _W)
    mech = rolling_prior(con_race, "constructor_key", ["_mech", "_one"], (config.MECH_WINDOW,))
    for k in _W:
        con[f"con_podium_rate_last{k}"] = con[f"_podium_sum{k}"] / con[f"_one_sum{k}"]
        con[f"con_mean_pos_last{k}"] = con[f"_pos_sum{k}"] / con[f"_one_sum{k}"]
    mech[MECH_FEATURE] = mech[f"_mech_sum{config.MECH_WINDOW}"] / mech[f"_one_sum{config.MECH_WINDOW}"]
    con = con.merge(mech[["constructor_key", "raceId", MECH_FEATURE]], on=["constructor_key", "raceId"],
                    how="left", validate="one_to_one")
    con = con.rename(columns={"races_before": "con_races_before"})
    keep = ["constructor_key", "raceId", "con_races_before", MECH_FEATURE] + [
        c for c in con.columns if c.startswith(("con_podium", "con_mean"))]
    df = df.merge(con[keep], on=["constructor_key", "raceId"], how="left", validate="many_to_one")
    _require_rows(df, n, "join constructor form")

    df = df.sort_values(["year", "round", "driverId"], kind="stable").reset_index(drop=True)
    df["split"] = assign_split(df["year"])
    meta = df[META_COLUMNS].copy()
    X = df[FEATURES].astype(float)
    y = (df["positionOrder"] <= PODIUM_MAX_POSITION).astype(int).rename("podium")

    if meta.duplicated(KEY).any():
        raise AssertionError("modeling table is not unique on (raceId, driverId)")
    forbidden = set(X.columns) & set(config.FORBIDDEN_CURRENT_RACE_COLUMNS)
    if forbidden or y.name in X.columns:
        raise AssertionError(f"forbidden columns in X: {sorted(forbidden)}")
    return meta, X, y
