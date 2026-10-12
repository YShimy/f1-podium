"""Generate notebooks/milestone1.ipynb.

This script is the source of truth for the notebook: edit the cells here, never
the .ipynb by hand (hand edits are overwritten on the next build).

Usage (from the repo root, with the venv's Python):
    python scripts/build_notebook.py [output.ipynb]
Then execute it so the outputs are saved (venv's Scripts folder first on PATH):
    jupyter nbconvert --to notebook --execute --inplace notebooks/milestone1.ipynb
"""

import sys
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

DEFAULT_OUT = Path(__file__).resolve().parents[1] / "notebooks" / "milestone1.ipynb"
out = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUT
md, code = new_markdown_cell, new_code_cell

cells = [
    md("""# F1 Podium Prediction — Milestone 1

> **Generated file: do not edit this notebook by hand.** The source of truth is
> `scripts/build_notebook.py`. Edit the cells there, then rebuild and re-execute the notebook
> (see that script's docstring). Hand edits to this `.ipynb` are overwritten on the next build.

Goal: for each driver in each race, predict whether they finish on the podium
(`results.positionOrder <= 3`), using only information available after qualifying.

Sections: **1** Setup, **2** Data Loading, **3** Raw Data Audit, **4** Audit follow-ups,
**5** Findings → open decisions, **6** Cleaning. Sections 3–4 only *measure* the raw data;
cleaning happens only in section 6, following decisions recorded in the notebook.
Reusable code lives in `src/f1podium/`."""),

    md("""## 1. Setup

Make the `src/` folder importable. We search upwards from the current folder so the
same cell works when run from `notebooks/`, from the repo root, or on Kaggle
(where `src/` must be provided alongside the notebook)."""),
    code("""import sys
from pathlib import Path

def _add_src_to_path() -> Path:
    for base in [Path.cwd(), *Path.cwd().parents]:
        if (base / "src" / "f1podium").is_dir():
            src = base / "src"
            if str(src) not in sys.path:
                sys.path.insert(0, str(src))
            return src
    raise FileNotFoundError("Could not find src/f1podium above " + str(Path.cwd()))

SRC_DIR = _add_src_to_path()

%load_ext autoreload
%autoreload 2"""),
    code("""import numpy as np
import pandas as pd
from IPython.display import display

from f1podium import config
from f1podium.clean import (
    CLEAN_STEP_DUPLICATE,
    CLEAN_STEP_INDY500,
    CLEAN_STEP_NON_STARTER,
    INDY500_CIRCUIT_REF,
    INDY500_YEARS,
    NON_START_STATUSES,
    STEP_NON_STARTER,
    WITHDREW_STATUS,
    clean_results,
)
from f1podium import plots
from f1podium.eda import rate_by, with_podium
from f1podium.load import load_all
from f1podium.validate import (
    find_duplicates,
    missing_summary,
    run_foreign_key_checks,
    run_primary_key_checks,
)

pd.set_option("display.max_rows", 250)
pd.set_option("display.max_columns", 50)
pd.set_option("display.width", 200)

np.random.seed(config.RANDOM_SEED)

print("Python     :", sys.version.split()[0], "|", sys.executable)
print("pandas     :", pd.__version__)
print("src        :", SRC_DIR)
print("raw data   :", config.RAW_DIR)
print("split      : train <=", config.TRAIN_END_YEAR, "| val", config.VAL_YEARS,
      "| test >=", config.TEST_START_YEAR)
print("seed       :", config.RANDOM_SEED)"""),

    md("""## 2. Data Loading

`load_all()` reads the 14 CSVs with an explicit schema per table:
`\\N` (and empty fields) become missing; IDs/counts use nullable `Int64`;
text uses `string`; dates are parsed strictly with `%Y-%m-%d`.
The loader raises if a file's columns differ from the declared schema."""),
    code("""tables = load_all()

load_summary = pd.DataFrame(
    {
        "rows": {n: len(df) for n, df in tables.items()},
        "columns": {n: df.shape[1] for n, df in tables.items()},
        "memory_MB": {n: round(df.memory_usage(deep=True).sum() / 1e6, 2) for n, df in tables.items()},
    }
).rename_axis("table")
display(load_summary)"""),
    md("Declared dtypes of the three tables we will use most:"),
    code("""for name in ["races", "results", "qualifying"]:
    print(f"--- {name}")
    print(tables[name].dtypes.to_string(), end="\\n\\n")"""),
    code("""# Short aliases used throughout the audit
races = tables["races"]
results = tables["results"]
qualifying = tables["qualifying"]

def with_year(df: pd.DataFrame) -> pd.DataFrame:
    \"\"\"Attach races.year, checking the join neither drops nor duplicates rows.\"\"\"
    out = df.merge(races[["raceId", "year"]], on="raceId", how="left", validate="many_to_one")
    assert len(out) == len(df), "join changed the row count"
    assert out["year"].notna().all(), "some rows have a raceId not in races"
    return out"""),

    md("""## 3. Raw Data Audit

### 3.1 Missing values per column

Count and percentage of missing values for every column of every table
(after treating `\\N` and empty fields as missing)."""),
    code("""missing = missing_summary(tables)
display(missing.sort_values(["table", "pct_missing"], ascending=[True, False]).reset_index(drop=True))"""),
    md("Only the columns that have at least one missing value, worst first:"),
    code("""display(
    missing[missing["n_missing"] > 0]
    .sort_values("pct_missing", ascending=False)
    .reset_index(drop=True)
)"""),

    md("""### 3.2 Primary keys

A key passes if it has no nulls and no repeated values. Surrogate ids (e.g. `raceId`)
are listed first, then natural/composite keys, which say more about the data.
`results (raceId, driverId)` is **expected to fail**: early seasons have shared drives."""),
    code("""pk_results = run_primary_key_checks(tables)
display(pk_results)
print(f"{(~pk_results['passed']).sum()} of {len(pk_results)} primary-key checks failed")"""),

    md("""### 3.3 Foreign keys

For each relationship: how many child rows have a null key, and how many point to an
id that does not exist in the parent table."""),
    code("""fk_results = run_foreign_key_checks(tables)
display(fk_results)
print(f"{(~fk_results['passed']).sum()} of {len(fk_results)} foreign-key checks failed")"""),

    md("""### 3.4 Duplicate `(raceId, driverId)` rows in `results`, by year

Our modelling grain is one row per `(raceId, driverId)`. Here we count, per season,
the rows that violate it. Nothing is dropped; the rule for handling them is an open decision."""),
    code("""dups = with_year(find_duplicates(results, ["raceId", "driverId"]))

dup_by_year = pd.DataFrame({
    "n_duplicated_pairs": dups.drop_duplicates(["raceId", "driverId"]).groupby("year").size(),
    "n_rows_involved": dups.groupby("year").size(),
    "n_races_affected": dups.groupby("year")["raceId"].nunique(),
})
dup_by_year.loc["Total"] = dup_by_year.sum()
display(dup_by_year)"""),
    md("Example duplicated entries (same driver, same race, more than one row):"),
    code("""example = dups.merge(
    tables["drivers"][["driverId", "driverRef"]], on="driverId", how="left", validate="many_to_one"
).merge(
    tables["constructors"][["constructorId", "constructorRef"]], on="constructorId", how="left", validate="many_to_one"
)
display(example[[
    "year", "raceId", "driverId", "driverRef", "constructorRef", "number",
    "grid", "positionText", "positionOrder", "points", "laps", "statusId",
]].head(12))"""),

    md("""### 3.5 Qualifying data: first year and coverage per season

Two coverage measures per season:

- **races covered**: share of races with *any* qualifying rows;
- **entries covered**: share of `results` rows (driver–race entries) whose
  `(raceId, driverId)` also appears in `qualifying`. This is what matters for our grain:
  it is the share of model rows that would have qualifying features."""),
    code("""quali_race_ids = set(qualifying["raceId"].dropna())
first_quali_year = int(races.loc[races["raceId"].isin(quali_race_ids), "year"].min())
print("First season with qualifying data:", first_quali_year)

quali_pairs = qualifying[["raceId", "driverId"]].drop_duplicates().assign(has_quali=True)
entries = with_year(results[["raceId", "driverId"]]).merge(
    quali_pairs, on=["raceId", "driverId"], how="left", validate="many_to_one"
)
assert len(entries) == len(results), "join changed the row count"
entries["has_quali"] = entries["has_quali"].fillna(False).astype(bool)

races_with_results = set(results["raceId"])
season = races.assign(
    has_results=races["raceId"].isin(races_with_results),
    has_quali=races["raceId"].isin(quali_race_ids),
)
quali_cov = pd.DataFrame({
    "n_races": season.groupby("year").size(),
    "n_races_with_results": season.groupby("year")["has_results"].sum(),
    "n_races_with_quali": season.groupby("year")["has_quali"].sum(),
    "n_results_rows": entries.groupby("year").size(),
    "n_results_rows_with_quali": entries.groupby("year")["has_quali"].sum(),
}).fillna(0).astype(int)
quali_cov["pct_races_covered"] = (100 * quali_cov["n_races_with_quali"] / quali_cov["n_races"]).round(1)
quali_cov["pct_entries_covered"] = (
    100 * quali_cov["n_results_rows_with_quali"] / quali_cov["n_results_rows"].replace(0, np.nan)
).round(1)
display(quali_cov)"""),

    md("""### 3.6 `grid = 0` by decade

`grid = 0` has no single documented meaning (e.g. pit-lane start vs. did not start).
We only count it here; how to treat it is an open decision."""),
    code("""grid = with_year(results[["raceId", "grid"]])
grid["decade"] = (grid["year"] // 10 * 10).astype(int)
grid_by_decade = grid.groupby("decade").agg(
    n_results_rows=("grid", "size"),
    n_grid_null=("grid", lambda s: int(s.isna().sum())),
    n_grid_zero=("grid", lambda s: int((s == 0).sum())),
)
grid_by_decade["pct_grid_zero"] = (100 * grid_by_decade["n_grid_zero"] / grid_by_decade["n_results_rows"]).round(2)
display(grid_by_decade)"""),

    md("""## 4. Audit follow-ups

Deeper looks at the three open questions raised by the audit. **Investigation only:**
nothing is cleaned, dropped or recoded in the data; any grouping below exists only to
make a table readable."""),

    md("""### 4.1 What do `grid = 0` rows look like?

Two cross-tabs of the `grid = 0` rows from `results`, by decade:

- **× status**: the race outcome recorded in `status` (e.g. *Did not qualify*, *Engine*, *Finished*).
- **× positionText**: Ergast's result code. Letters are non-classifications
  (`R` retired, `D` disqualified, `E` excluded, `W` withdrawn, `F` failed to qualify,
  `N` not classified). Numbers are classified finishing positions; to keep the table
  narrow they are grouped into *classified P1–3* and *classified P4+*.

If `grid = 0` mostly means "never started", we expect it to sit with *Did not qualify* /
`F` / `W`. If it means "started from the pit lane", we expect normal outcomes
(finishes, retirements)."""),
    code("""POS_TEXT_LABELS = {
    "R": "R (retired)",
    "D": "D (disqualified)",
    "E": "E (excluded)",
    "W": "W (withdrawn)",
    "F": "F (failed to qualify)",
    "N": "N (not classified)",
}

def position_text_group(pt: str) -> str:
    \"\"\"Readable group for a positionText value (display only).\"\"\"
    if pt.isdigit():
        return "classified P1-3" if int(pt) <= 3 else "classified P4+"
    return POS_TEXT_LABELS.get(pt, pt)

grid0 = with_year(results.loc[results["grid"] == 0, ["raceId", "statusId", "positionText"]])
grid0["decade"] = (grid0["year"] // 10 * 10).astype(int)
grid0 = grid0.merge(tables["status"], on="statusId", how="left", validate="many_to_one")
assert grid0["status"].notna().all(), "grid = 0 row with unknown statusId"
grid0["positionText_group"] = grid0["positionText"].map(position_text_group)
print("grid = 0 rows:", len(grid0))

def crosstab_by_decade(row_col: str, top_n: int | None = None) -> pd.DataFrame:
    \"\"\"Rows sorted by total; optionally keep the top_n rows and fold the rest into 'Other'.\"\"\"
    ct = pd.crosstab(grid0[row_col], grid0["decade"], margins=True, margins_name="Total")
    body = ct.drop(index="Total").sort_values("Total", ascending=False)
    if top_n is not None and len(body) > top_n:
        rest = body.iloc[top_n:]
        other = rest.sum().rename(f"Other ({len(rest)} statuses)")
        body = pd.concat([body.iloc[:top_n], other.to_frame().T])
    return pd.concat([body, ct.loc[["Total"]]])"""),
    md("**`grid = 0` × status**: the top 15 statuses, with the rest folded into *Other*. The last row is the column total:"),
    code("display(crosstab_by_decade(\"status\", top_n=15))"),
    md("**`grid = 0` × positionText group:**"),
    code("display(crosstab_by_decade(\"positionText_group\"))"),
    md("""**What this shows.**

- **Before 2000, `grid = 0` almost always means "did not start".** Of the 1,638 rows,
  1,348 are `F` (failed to qualify): 1,013 *Did not qualify* plus 331 *Did not prequalify*.
  Another 203 are `W` (withdrawn). These cars never took the start. Prequalifying shows up only
  in the 1970s–90s, when more cars entered than could race.
- **From the 2010s on, it mostly means "started from the pit lane".** The statuses become
  *Finished*, *+1 Lap* and mechanical retirements, so these cars raced. All 60 classified
  `grid = 0` rows fall in 2015–2024, and the best finish is **P5**. No `grid = 0` entry in this
  dataset reached the podium.
- **Some rows are still unexplained (Unknown).** 18 *Accident* rows from the 1950s–1990s have
  `grid = 0`. These may be accidents in practice or qualifying; the cause is not investigated here.
- **Leakage note.** `status` and `positionText` are post-race columns, so they cannot be model
  features. They are used here only to understand what `grid = 0` means. Failing to qualify is
  known *after qualifying*, which is our prediction cutoff, so deciding how to treat those rows
  remains an open decision rather than a leakage problem."""),

    md("""#### 4.1b Non-start statuses: `grid = 0` vs `grid > 0`

The other direction: start from statuses that *sound* like a non-start and check their grid value.
`laps_gt0` counts rows that completed at least one lap, which is direct evidence that the car raced.

The candidate list is for investigation only; it is **not** a non-starter definition. It includes
*Not classified* to test it. In F1 terminology, *not classified* means the car started but covered
too little of the race distance to be classified. `laps` is a post-race column, used here as evidence only."""),
    code("""NON_START_CANDIDATES = [
    "Did not qualify", "Did not prequalify", "107% Rule", "Withdrew", "Not classified",
    "Excluded", "Injury", "Injured", "Driver unwell", "Safety concerns", "Underweight",
]
missing_statuses = set(NON_START_CANDIDATES) - set(tables["status"]["status"])
assert not missing_statuses, f"not in status table: {missing_statuses}"

cand = (
    results[["raceId", "statusId", "grid", "laps"]]
    .merge(tables["status"], on="statusId", how="left", validate="many_to_one")
    .query("status in @NON_START_CANDIDATES")
)
cand["grid_group"] = np.where(cand["grid"] > 0, "grid>0", "grid=0")
non_start_split = (
    cand.groupby(["status", "grid_group"])
    .agg(rows=("laps", "size"), laps_gt0=("laps", lambda s: int((s > 0).sum())))
    .unstack("grid_group", fill_value=0)
    .reindex(NON_START_CANDIDATES)
    .fillna(0)
    .astype(int)
)
non_start_split.columns = [f"{grid}: {measure}" for measure, grid in non_start_split.columns]
non_start_split = non_start_split[["grid=0: rows", "grid=0: laps_gt0", "grid>0: rows", "grid>0: laps_gt0"]]
display(non_start_split)"""),
    md("Where the `Withdrew` rows with `grid > 0` come from (the top 5 races):"),
    code("""withdrew_started_grid = with_year(
    results.loc[(results["statusId"].isin(tables["status"].query("status == 'Withdrew'")["statusId"]))
                & (results["grid"] > 0), ["raceId", "laps"]]
).merge(races[["raceId", "name"]], on="raceId", how="left", validate="many_to_one")
display(
    withdrew_started_grid.groupby(["year", "name"])
    .agg(rows=("laps", "size"), laps_gt0=("laps", lambda s: int((s > 0).sum())))
    .sort_values("rows", ascending=False)
    .head(5)
)"""),
    md("""**What this shows.**

- **Most non-start statuses go with `grid = 0`, but not all of them.** *Did not prequalify* is 331 rows, all with
  `grid = 0`. *Did not qualify* has 1,013 rows with `grid = 0` and 12 with `grid > 0`; one of those 12 even
  completed a lap.
- **`Withdrew` covers three different situations:**
  - 164 rows with `grid = 0`: never took a grid slot.
  - 61 rows with `grid > 0` and no laps: qualified, then did not start. 14 of these are the
    2005 US GP, where most of the field withdrew after the formation lap.
  - 23 rows with `grid > 0` and at least one lap: started the race, then withdrew.
- **`Not classified` is not a non-start.** All 172 rows have `grid > 0` and at least one lap.
- **Other candidates are small and mixed.**
  - *107% Rule:* 7 of its 9 rows have `grid > 0`, but none completed a lap.
  - *Driver unwell:* 9 of its 10 rows completed laps, so it mostly means "started, then stopped".
  - *Excluded* and *Injury:* no `grid = 0` row completed a lap, but one `grid > 0` row each did.
- **So neither signal is enough on its own.** `grid = 0` alone and status alone each misclassify
  some entries. Any non-starter definition has to say which signals it uses, and whether those
  signals are known at the prediction cutoff."""),

    md("""### 4.2 The 85 duplicate `(raceId, driverId)` pairs

One row per duplicated pair, describing its rows:

- `n_rows`: how many `results` rows the driver has in that race;
- `n_rows_grid_pos`: how many of those rows have `grid > 0` (i.e. a real starting slot);
- `positionTexts`: each row's `positionText`, in `positionOrder` order;
- `constructors_differ`: whether the rows are for different constructors.

Then the pairs are summarised as **counts per pattern**. For the pattern,
`positionText` is reduced to *classified* vs. its letter code, so that e.g. `4 / R` and
`7 / R` count as the same pattern (`R + classified`)."""),
    code("""dup_rows = (
    with_year(find_duplicates(results, ["raceId", "driverId"]))
    .merge(tables["drivers"][["driverId", "driverRef"]], on="driverId", how="left", validate="many_to_one")
    .sort_values(["year", "raceId", "driverId", "positionOrder"])
)
dup_rows["pt_kind"] = dup_rows["positionText"].map(lambda pt: "classified" if pt.isdigit() else pt)

dup_pairs = (
    dup_rows.groupby(["raceId", "driverId"], sort=False)
    .agg(
        year=("year", "first"),
        driverRef=("driverRef", "first"),
        n_rows=("resultId", "size"),
        n_rows_grid_pos=("grid", lambda s: int((s > 0).sum())),
        positionTexts=("positionText", lambda s: " / ".join(s)),
        n_constructors=("constructorId", "nunique"),
        result_pattern=("pt_kind", lambda s: " + ".join(sorted(s))),
    )
    .reset_index()
)
dup_pairs["constructors_differ"] = dup_pairs["n_constructors"] > 1
indy_race_ids = set(races.loc[races["name"] == "Indianapolis 500", "raceId"])
dup_pairs["indy500"] = dup_pairs["raceId"].isin(indy_race_ids)
assert len(dup_pairs) == dup_rows[["raceId", "driverId"]].drop_duplicates().shape[0]
print("duplicated pairs:", len(dup_pairs), "| rows involved:", len(dup_rows),
      "| pairs from the Indianapolis 500:", int(dup_pairs["indy500"].sum()))
display(dup_pairs[[
    "year", "raceId", "driverRef", "n_rows", "n_rows_grid_pos",
    "positionTexts", "constructors_differ", "indy500",
]])"""),
    md("**Counts per pattern** (`years` = first–last season the pattern occurs):"),
    code("""dup_patterns = (
    dup_pairs.groupby(["n_rows", "n_rows_grid_pos", "constructors_differ", "result_pattern"])
    .agg(
        n_pairs=("year", "size"),
        n_indy500=("indy500", "sum"),
        years=("year", lambda y: f"{y.min()}-{y.max()}"),
    )
    .reset_index()
    .sort_values("n_pairs", ascending=False, ignore_index=True)
)
assert dup_patterns["n_pairs"].sum() == len(dup_pairs)
display(dup_patterns)"""),
    md("""**What this shows.**

- **Each row is one car the driver drove, and it carries that car's grid slot and result.**
  This was confirmed on the 1951 French GP (raceId 828). Fangio has one row for car 8, grid 7,
  **1st**: Fagioli's car, which Fangio took over and won in. He has a second row for his own
  car 24, grid 1, 11th. Fagioli has rows for the same two cars. So in shared drives, `grid`
  describes the *car*, not where that driver started.
- **Most duplicates are classic 1950s shared drives.** 82 of the 85 pairs have `grid > 0` in every
  row. The most common pattern is *R + classified* with the same constructor (46 pairs): the
  driver's own car retired, and they finished the race in a teammate's car. In 8 pairs, both
  cars the driver drove were classified.
- **3 pairs are not shared drives.** They have `grid = 0` in both rows: *F + F* in 1958 and 1978,
  and *W + W* in 1962. The driver was entered in two cars and started neither. These overlap
  with the non-starter question in 4.1.
- **21 of the 85 pairs come from the Indianapolis 500** (1950, 1953, 1954). In about half of them
  (11 of 21) the constructors differ, because drivers relieved each other across teams. This links the shared-drive
  rule to the open Indy 500 exclusion decision.
- **Why it matters for the target.** With shared drives, one driver can have both a podium row
  and a non-podium row in the same race (Fangio 1951 above). Any rule for these rows has to say
  what `podium` means for such a driver. That is an open decision and is not settled here."""),

    md("""### 4.3 Qualifying coverage per season, 1990–2024

Share of race entries (`results` rows) that have a matching `qualifying` row: the
`pct_entries_covered` column from 3.5, plotted. Dashed lines mark the planned
train / validation / test boundaries. (Race-level coverage is almost identical and is
in the 3.5 table.)"""),
    code("""import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

INK, INK_MUTED, SERIES, SURFACE = "#0b0b0b", "#52514e", "#2a78d6", "#fcfcfb"

cov = quali_cov.loc[1990:2024, "pct_entries_covered"]
years, pct = cov.index.astype(int), cov.astype(float).to_numpy()

fig, ax = plt.subplots(figsize=(10, 4.2), facecolor=SURFACE)
ax.set_facecolor(SURFACE)
ax.plot(years, pct, color=SERIES, linewidth=2, marker="o", markersize=5,
        markeredgecolor=SURFACE, markeredgewidth=1.2, zorder=3)

# Planned split boundaries
for x in (config.TRAIN_END_YEAR + 0.5, config.TEST_START_YEAR - 0.5):
    ax.axvline(x, color=INK_MUTED, linewidth=1, linestyle="--", zorder=1)
for x, label in [((1990 + config.TRAIN_END_YEAR) / 2, "train"),
                 (sum(config.VAL_YEARS) / 2, "val"),
                 ((config.TEST_START_YEAR + 2024) / 2, "test")]:
    ax.text(x, 103, label, ha="center", va="bottom", color=INK_MUTED, fontsize=9)

# Selective direct labels: first year with data and the lowest season after it
first = int(years[pct > 0][0])
gap_years = cov.loc[first:2002]
low = int(gap_years.idxmin())
ax.annotate(f"{first}: first data", (first, cov[first]), xytext=(-8, 0),
            textcoords="offset points", ha="right", va="center", color=INK_MUTED, fontsize=9)
ax.annotate(f"{low}: {cov[low]:.1f}%", (low, cov[low]), xytext=(8, -4),
            textcoords="offset points", color=INK_MUTED, fontsize=9)

ax.set_xlim(1989.5, 2024.5)
ax.set_ylim(0, 110)
ax.set_yticks([0, 25, 50, 75, 100])
ax.yaxis.set_major_formatter(PercentFormatter(100))
ax.set_xticks(range(1990, 2025, 5))
ax.grid(axis="y", color="#e4e3df", linewidth=0.8, zorder=0)
for side in ("top", "right", "left"):
    ax.spines[side].set_visible(False)
ax.spines["bottom"].set_color(INK_MUTED)
ax.tick_params(colors=INK_MUTED, length=0, labelsize=9)
ax.set_title("Race entries with a qualifying record, by season (1990-2024)",
             loc="left", color=INK, fontsize=12, pad=18)
ax.set_ylabel("Entries covered", color=INK_MUTED, fontsize=9)
fig.tight_layout()
plt.show()"""),
    md("""**What this shows.**

- **There are four phases.** There is no qualifying data before 1994. 1994–1995 is nearly
  complete. **1996–2002 is patchy**, as low as 5.9% of entries in 2001. **From 2003 on**, coverage
  is about 99–100%.
- **Validation (2020–21) and test (2022+) are fully covered.** The coverage gap affects only the
  training years.
- **The training window and missing-qualifying strategy are linked.** A window starting in 2003
  has near-complete qualifying data. An earlier start brings many rows with no qualifying times.
  `results.grid` has no nulls in any year (3.6), so it is always available as a pre-race
  position signal.
- **The small dips in 2011–2019 (~99%)** are a handful of entries without a qualifying row.
  Their cause is Unknown and was not investigated here."""),

    md("""## 5. Findings → open decisions

Decisions are labelled D1, D2, … and recorded in full in section 6.

| Finding | Evidence (section) | Open decision | Rule | Status |
|---|---|---|---|---|
| 85 duplicated `(raceId, driverId)` pairs, mostly 1950s shared drives | 3.4, 4.2 | **D1** Shared-drive rule | Rule B: target-blind heuristic (started row, lowest grid, lowest resultId; see 6.3) | **Decided** |
| 11 Indianapolis 500 races (1950–1960); they also hold 21 of the 85 duplicate pairs | 4.2 | **D2** Indy 500 exclusion | Exclude races at the Indianapolis circuit in 1950–1960 (see 6.1) | **Decided** |
| About 1,500 entries that never started (DNQ, DNPQ, 107% rule, withdrawn without a grid slot) | 4.1, 4.1b | **D3** Non-starters | Exclude by status, plus grid for *Withdrew* (see 6.2) | **Decided** |
| … | 3.5, 4.3 | Missing-qualifying strategy / training window | … | Pending |
| `grid = 0` means "did not start" before 2000 and mostly "pit-lane start" after 2010 | 3.6, 4.1 | Meaning of `grid = 0` **as a feature** | … | Pending |"""),

    md("""## 6. Cleaning

Cleaning steps implement the decisions recorded below. Each step is a pure function in
`src/f1podium/clean.py` that returns a new table plus a log; the raw tables are never modified.

**Pipeline order for `results`:** D2 Indianapolis 500 exclusion → D3 non-starter filter → D1 duplicate resolution.

- **D2 runs first** because it removes whole races, including 21 of the duplicate pairs. Those pairs
  then never reach the duplicate rule.
- **D3 runs before D1** so that a did-not-qualify row can never be the row D1 keeps, and so that pairs
  made only of non-starters disappear entirely. After D3, every remaining duplicate pair consists of
  cars that took a grid slot.

The whole pipeline is one call, `clean_results`, which fixes this order (a unit test fails if the order changes)
and itself raises if `(raceId, driverId)` is not unique at the end. It returns the cleaned table and a
**combined log** with one row per removed `results` row, labelled with the step that removed it. Sections
6.1–6.3 below read their per-step views from that log."""),
    code("""results_clean, clean_log = clean_results(results, races, tables["circuits"], tables["status"])

print(f"results rows: {len(results)} -> {len(results_clean)} (removed {len(clean_log)})")
display(clean_log["step"].value_counts().rename_axis("step").to_frame("rows_removed"))

def log_rows(step: str) -> pd.DataFrame:
    \"\"\"Rows of the combined log removed by one step.\"\"\"
    return clean_log[clean_log["step"] == step]"""),

    md("""### 6.1 D2: Indianapolis 500 exclusion

#### Decision record

**Problem.** From 1950 to 1960 the Indianapolis 500 counted towards the drivers' championship, so it appears in
`races` and `results`. It also accounts for 21 of the 85 duplicate pairs (4.2): relief drivers moved across teams.

**Options considered**

| Option | Rule | Assessment |
|---|---|---|
| A | Keep the races | Mixes in a race that does not behave like a Grand Prix (see *Why*). |
| B | Keep them, with an "is Indy 500" indicator | Lets a model separate them, but 11 races are too few to learn a separate pattern from. |
| **C** | **Exclude all results of these races** | Chosen. |

**Choice: C.** Exclude every `results` row of a race held at the circuit with `circuitRef == "indianapolis"`
in 1950–1960 (inclusive).

**Why** *(drafted for your review)*. The Indy 500 was run to American (AAA/USAC) rules, with its own qualifying format
and a field almost entirely separate from the Grand Prix regulars. Its entries therefore say little about the
podium chances of an F1 field after qualifying, which is what we predict.

**How it is identified.** It is identified by circuit **and** year range, not by name. The same circuit later hosted the
United States Grand Prix (2000–2007), which is an ordinary F1 race and is kept. The table below shows every
race held at the circuit. An assertion checks that the selection equals exactly the races named
"Indianapolis 500"."""),
    code("""indy_circuit = tables["circuits"].loc[
    tables["circuits"]["circuitRef"] == INDY500_CIRCUIT_REF, ["circuitId", "circuitRef", "name"]
]
display(indy_circuit)

at_indy = races[races["circuitId"].isin(indy_circuit["circuitId"])]
display(
    at_indy.groupby("name")
    .agg(first_year=("year", "min"), last_year=("year", "max"), n_races=("raceId", "size"))
    .sort_values("first_year")
)
print("Excluded year range (inclusive):", INDY500_YEARS)"""),
    code("""indy_rows = log_rows(CLEAN_STEP_INDY500)
indy_log = (
    indy_rows.groupby(["raceId", "year"]).size().rename("n_rows_excluded").reset_index()
    .merge(races[["raceId", "name"]], on="raceId", how="left", validate="one_to_one")
    .sort_values("year", ignore_index=True)
)

named_indy500 = set(races.loc[races["name"] == "Indianapolis 500", "raceId"])
assert set(indy_log["raceId"]) == named_indy500, "selection != races named 'Indianapolis 500'"
assert len(indy_rows) == results["raceId"].isin(named_indy500).sum(), "not every Indy 500 row was removed"

print(f"D2 removed {len(indy_rows)} rows from {len(indy_log)} races")
display(indy_log[["raceId", "year", "name", "n_rows_excluded"]])"""),

    md("""### 6.2 D3: Non-starter filter

#### Decision record

**Problem.** About 1,500 `results` rows describe entries that never took the start (4.1, 4.1b). We predict podiums for
the cars that line up after qualifying, so these rows are not part of the population we predict for.

**Choice.** Exclude a row when:
- its status is *Did not qualify*, *Did not prequalify* or *107% Rule*, whatever its grid; **or**
- its status is *Withdrew* **and** `grid == 0`.

**Why status is acceptable here.** CLAUDE.md lists `status` as a post-race column, and it is never used as a
**feature**. Here it is used only to define *which rows are in the population*, and only through values that are
settled by qualifying:
- not qualifying, not prequalifying and failing the 107% rule are all decided in qualifying, so they are known at our cutoff;
- *Withdrew* with `grid = 0` means the car never took a grid slot *(Assumption: the withdrawal is known before the start)*.

**Deliberately kept:**
- **Not classified:** these cars started (4.1b: all 172 rows completed laps).
- **Pit-lane starts:** `grid = 0` with any other status, mostly 2010 onwards.
- **Withdrew with `grid > 0`:** the car qualified and withdrew later. That is not known at the cutoff in general
  (e.g. 2005 US GP), so these remain as real entries; their label is non-podium.

**Never used for selection:** `laps` or any other post-race column. The diagnostic in 6.4 looks at `laps`
afterwards, only to show what remains."""),
    code("""non_start_log = (
    log_rows(CLEAN_STEP_NON_STARTER)
    .merge(results[["resultId", "statusId", "grid"]], on="resultId", how="left", validate="one_to_one")
    .merge(tables["status"], on="statusId", how="left", validate="many_to_one")
)
assert set(non_start_log["status"]) <= set(NON_START_STATUSES) | {WITHDREW_STATUS}
assert (non_start_log.loc[non_start_log["status"] == WITHDREW_STATUS, "grid"] == 0).all()

print(f"D3 removed {len(non_start_log)} rows")
display(non_start_log["reason"].value_counts().rename_axis("reason").to_frame("n_rows"))"""),
    md("Excluded rows by status × decade:"),
    code("""non_start_log["decade"] = (non_start_log["year"] // 10 * 10).astype(int)
display(pd.crosstab(non_start_log["status"], non_start_log["decade"], margins=True, margins_name="Total"))"""),

    md("""### 6.3 D1: Duplicate `(raceId, driverId)` rows in `results`

#### Decision record

**Problem.** In the raw data, 85 driver–race pairs have more than one `results` row (176 rows, 1950–1978; see 3.4 and 4.2).
Our grain needs exactly one row per driver per race. After D2 removes the Indianapolis 500 (21 pairs) and D3 removes
the pairs made only of non-starters (3 pairs), **61 pairs** reach this step.

**Options considered**

| Option | Rule | Assessment |
|---|---|---|
| A | Keep the row with the best `positionOrder` | Chooses rows **using the target**: a driver who took over a winning car gets the podium. This is label leakage and inflates podium rates. |
| **B** | **Keep one row per pair chosen by pre-race fields only** (`grid > 0`, then lowest grid, then lowest `resultId`) | Target-blind: the choice never looks at positions, points or status, so it cannot favour podium rows. |
| C | Drop every duplicated pair | Deletes real race entries, including genuine podiums, and leaves holes in those races. |
| D | Aggregate rows (e.g. best result, mean grid) | Creates a row that never existed: its features and label come from different cars. |

**Choice: B**, applied in this order:
1. Pairs where no row has `grid > 0` are non-starters. They are left untouched and logged as
   *resolved by non-starter filter*. Because D3 runs first, this step is expected never to be used; an assertion checks it.
2. Keep the row with `grid > 0`.
3. If several rows have `grid > 0`, keep the lowest grid.
4. Final deterministic tie-break: the lowest `resultId`.

**Why.** B is **not** "keep the car the driver started in"; the data cannot tell us that. **Every resolved
pair has more than one row with `grid > 0`.** Step 2 never decides a real pair (see the step counts below:
0 × *single started row*). In a shared drive, every row carries *that car's* grid slot, not the driver's.
So the lowest grid is a **target-blind heuristic for the driver's primary entry**: the
better-placed car they are listed with. What B guarantees is that the row is chosen without looking at the
outcome, unlike rule A, and is fully deterministic. It does not guarantee that the kept row is the car the
driver started in. *(Assumption: the lowest grid usually corresponds to the primary entry; Unknown how often.)*

**Known misattributions (accepted).** Here the heuristic keeps a car the driver took over, not the one they started in:
- **Fagioli, 1951 French GP (raceId 828).** He started car 8 from grid 7. Fangio took it over and won.
  Fagioli then drove Fangio's car 24 (grid 1, classified 11th). B keeps the car-24 row (resultId 20188, P11)
  instead of car 8 (resultId 19918, P1).
- **Brooks, 1957 British GP (raceId 780).** Moss took over the car Brooks started in (grid 3) and won.
  Brooks drove Moss's car (grid 1, classified 12th). B keeps resultId 20293 (P12) instead of 18806 (P1).

In both cases the label changes from podium to non-podium. The log column
`lowest_grid_differs_from_lowest_resultId` flags the pairs where the lowest grid and the lowest `resultId`
disagree (counted below). The lowest `resultId` is often the originally entered car, but it is only partial evidence,
so these flagged pairs are where further misattributions are most likely.

**Measured impact:** see the end of this section."""),
    code("""# One row per duplicated pair, rebuilt from the combined log (one log row per dropped results row).
dup_log = (
    log_rows(CLEAN_STEP_DUPLICATE)
    .groupby(["raceId", "driverId"], as_index=False)
    .agg(
        year=("year", "first"),
        n_rows=("resultId", lambda s: len(s) + 1),  # dropped rows + the kept row
        resolving_step=("resolving_step", "first"),
        kept_resultId=("kept_resultId", "first"),
        dropped_resultIds=("resultId", lambda s: sorted(int(x) for x in s)),
        lowest_grid_differs_from_lowest_resultId=("lowest_grid_differs_from_lowest_resultId", "first"),
    )
    .sort_values(["year", "raceId", "driverId"], ignore_index=True)
)

print("Pairs per resolving step:")
display(dup_log["resolving_step"].value_counts().rename_axis("resolving_step").to_frame("n_pairs"))
display(dup_log)"""),
    md("""**Checks.** Every check below is an assertion, so the notebook stops if any of them fails.
`clean_results` already raises if the final table is not unique; these repeat the checks visibly:
- **Row count:** the cleaned table has exactly *raw − removed* rows.
- **Uniqueness:** `(raceId, driverId)` is **fully** unique.
- **No pending pairs:** the *non-starter* step of D1 was never needed, because D3 ran first.
- **Kept rows exist:** every pair's kept row is in the cleaned table."""),
    code("""n_dropped = int(dup_log["dropped_resultIds"].str.len().sum())
assert len(results_clean) == len(results) - len(clean_log), "row count != raw - removed"
assert not results_clean.duplicated(["raceId", "driverId"]).any(), "(raceId, driverId) is not unique"
assert not (dup_log["resolving_step"] == STEP_NON_STARTER).any(), "non-starter pairs reached D1"
assert dup_log["kept_resultId"].isin(results_clean["resultId"]).all(), "a kept row is missing"

print(f"D1 dropped {n_dropped} rows in {len(dup_log)} pairs")
print("All checks passed: (raceId, driverId) is unique.")"""),

    md("""#### Sensitivity check: rule B vs rule A (best `positionOrder`)

For each pair that rule B resolved, we also apply rule A (best `positionOrder`, with the lowest `resultId` as tie-break).
We then count how often the two rules keep a different row, and how often the **podium label**
(`positionOrder <= 3`) differs. This shows how much the label depends on the choice.
`positionOrder` is used here only to measure this; it plays no part in rule B itself."""),
    code("""# The rows D1 chose between: each pair's kept row plus its dropped rows.
pair_result_ids = set(dup_log["kept_resultId"]) | {r for ids in dup_log["dropped_resultIds"] for r in ids}
dup_rows_all = results[results["resultId"].isin(pair_result_ids)]
assert len(dup_rows_all) == dup_log["n_rows"].sum(), "pair rows do not match the log"

rule_a = (
    dup_rows_all.sort_values(["positionOrder", "resultId"])
    .groupby(["raceId", "driverId"]).head(1)
    [["raceId", "driverId", "resultId"]]
    .rename(columns={"resultId": "kept_A"})
)
compare = (
    dup_log[["year", "raceId", "driverId", "kept_resultId", "lowest_grid_differs_from_lowest_resultId"]]
    .rename(columns={"kept_resultId": "kept_B"})
    .merge(rule_a, on=["raceId", "driverId"], how="left", validate="one_to_one")
)
by_id = results.set_index("resultId")
for rule in ("A", "B"):
    kept = compare[f"kept_{rule}"]
    compare[f"grid_{rule}"] = kept.map(by_id["grid"])
    compare[f"positionOrder_{rule}"] = kept.map(by_id["positionOrder"])
    compare[f"podium_{rule}"] = compare[f"positionOrder_{rule}"] <= 3
assert compare[["kept_A", "positionOrder_A", "positionOrder_B"]].notna().all().all()

podium_differs = compare["podium_A"] != compare["podium_B"]
sensitivity = pd.Series({
    "pairs resolved by rule B": len(compare),
    "kept row differs (A vs B)": int((compare["kept_A"] != compare["kept_B"]).sum()),
    "podium label differs (A vs B)": int(podium_differs.sum()),
    "podium labels in these pairs, rule A": int(compare["podium_A"].sum()),
    "podium labels in these pairs, rule B": int(compare["podium_B"].sum()),
    "pairs flagged: lowest grid != lowest resultId": int(compare["lowest_grid_differs_from_lowest_resultId"].sum()),
}).to_frame("count")
display(sensitivity)"""),
    md("Pairs whose podium label depends on the rule:"),
    code("""display(
    compare.loc[podium_differs]
    .merge(tables["drivers"][["driverId", "driverRef"]], on="driverId", how="left", validate="many_to_one")
    [["year", "raceId", "driverRef", "kept_B", "grid_B", "positionOrder_B",
      "kept_A", "grid_A", "positionOrder_A", "lowest_grid_differs_from_lowest_resultId"]]
    .reset_index(drop=True)
)"""),
    md("""**Measured impact** (on the D3 output, i.e. after the Indy 500 and non-starters are removed).

- **Rows.** 24,825 → 24,761: 64 rows dropped across 61 pairs. `(raceId, driverId)` is now fully unique.
  For comparison, on raw `results` the same rule would see 85 pairs. 21 of those were Indy 500 pairs (removed by D2),
  and 3 were made only of non-starters (removed by D3).
- **Steps used.** 60 pairs were resolved by *lowest grid* and 1 by the *lowest resultId* tie-break. *Single started row*
  and *non-starter* were never used. Every pair has `grid > 0` on all its rows, which is why step 3 does almost all the work.
- **Sensitivity to the rule.** Rules A and B keep a different row in 44 of 61 pairs, but the **podium label differs in
  17 pairs**. All 17 go the same way: rule A gives a podium and rule B does not (21 vs 4 podium labels in these pairs).
  So rule A would add 17 podium labels. All 17 are from 1950–1960, inside the training period; validation and test rows
  are unaffected.
- **Where the heuristic misattributes.** 16 of the 61 pairs are flagged (lowest grid ≠ lowest resultId). 4 of the 17 flips are
  flagged, including the two known misattributions (Fagioli 1951, Brooks 1957). This is accepted for now. A possible
  follow-up (Experiment) is to put the lowest-resultId tie-break before the lowest-grid one and re-measure."""),

    md("""### 6.4 Pipeline summary

Row counts after each step, with assertions that they chain exactly and that the final table has one row
per `(raceId, driverId)`."""),
    code("""step_order = [CLEAN_STEP_INDY500, CLEAN_STEP_NON_STARTER, CLEAN_STEP_DUPLICATE]
removed = clean_log["step"].value_counts().reindex(step_order, fill_value=0)
pipeline_counts = pd.DataFrame({
    "rows_removed": [0, *removed.tolist()],
    "rows": [len(results), *(len(results) - removed.cumsum()).tolist()],
}, index=pd.Index(["raw results", *[f"after {s}" for s in step_order]], name="step"))
display(pipeline_counts)

assert pipeline_counts["rows"].iloc[-1] == len(results_clean), "row counts do not chain to the cleaned table"
assert set(clean_log["step"]) <= set(step_order), "unexpected step label in the log"
assert not results_clean.duplicated(["raceId", "driverId"]).any(), "final table is not unique"
assert len(results) == len(tables["results"]) == 26759, "raw results table changed"
print(f"results_clean: {len(results_clean)} rows, one per (raceId, driverId).")"""),

    md("""#### Diagnostic: remaining rows with `laps == 0`

**Diagnostic only, not used to select rows.** `laps` is a post-race column. This table shows what is left in
`results_clean` among entries that completed no lap, so we can see whether the population looks right."""),
    code("""remaining_no_laps = (
    results_clean.loc[results_clean["laps"] == 0, ["statusId", "grid"]]
    .merge(tables["status"], on="statusId", how="left", validate="many_to_one")
)
remaining_no_laps["grid_group"] = np.where(remaining_no_laps["grid"] > 0, "grid>0", "grid=0")
no_laps_by_status = pd.crosstab(remaining_no_laps["status"], remaining_no_laps["grid_group"], margins=True, margins_name="Total")
no_laps_by_status = pd.concat([
    no_laps_by_status.drop(index="Total").sort_values("Total", ascending=False),
    no_laps_by_status.loc[["Total"]],
])
print(f"remaining rows with laps == 0: {len(remaining_no_laps)} of {len(results_clean)}")
display(no_laps_by_status.head(16))"""),
    md("""**What this shows** (diagnostic only; nothing here changes the selection).

- **995 of the 24,761 remaining rows completed no lap.** Most are real starts that ended on the first lap:
  *Collision* (353) and *Accident* (242) with `grid > 0`, plus mechanical failures (*Engine*, *Gearbox*, *Clutch*, …).
  These are entries with a correct non-podium label.
- **61 *Withdrew* rows with `grid > 0` remain by design (D3).** The car qualified and withdrew later, which is not known
  at our cutoff, so it stays as a non-podium entry. The 2005 US GP accounts for 14 of them.
- **56 rows have `grid = 0` and no laps** (1954–2023; 8 of them from 2000 onwards). They are mostly *Accident* (18),
  *Injury* (8), *Excluded* (6), *Disqualified* (6) and *Engine* (6). Some of these may be entries that never started
  (for example, an accident in practice). D3 does not catch them, because their status does not say "did not start".
  *Unknown:* how many of them actually took the start. This is noted here only; any change to D3 would be a new decision."""),

    md("""#### Diagnostic: the remaining `grid = 0` rows with no laps

**Diagnostic only, not used to select rows.** These are the 56 rows from the note above: `grid == 0` and `laps == 0`
in `results_clean`. Their statuses do not say "did not start", so D3 keeps them, but some look non-start-like.
They are grouped by status and by era. The era boundary is 2003, the first season with complete qualifying
data (4.3)."""),
    code("""grid0_no_laps = (
    results_clean.loc[(results_clean["grid"] == 0) & (results_clean["laps"] == 0),
                      ["resultId", "raceId", "driverId", "statusId", "grid", "laps"]]
    .merge(tables["status"], on="statusId", how="left", validate="many_to_one")
    .merge(races[["raceId", "year", "name"]], on="raceId", how="left", validate="many_to_one")
    .merge(tables["drivers"][["driverId", "driverRef"]], on="driverId", how="left", validate="many_to_one")
)
grid0_no_laps["era"] = np.where(grid0_no_laps["year"] < 2003, "before 2003", "2003+")
print(f"remaining rows with grid == 0 and laps == 0: {len(grid0_no_laps)}")

by_status_era = pd.crosstab(grid0_no_laps["status"], grid0_no_laps["era"], margins=True, margins_name="Total")
by_status_era = pd.concat([
    by_status_era.drop(index="Total").sort_values("Total", ascending=False),
    by_status_era.loc[["Total"]],
])[["before 2003", "2003+", "Total"]]
display(by_status_era)"""),
    md("All rows, by era and year:"),
    code("""display(
    grid0_no_laps.sort_values(["era", "year", "raceId"], ascending=[False, True, True])
    [["era", "year", "name", "driverRef", "status", "grid", "laps", "resultId"]]
    .reset_index(drop=True)
)"""),
    md("""**What this shows** (diagnostic only; nothing here changes the selection or decides anything).

- **51 of the 56 rows are from before 2003 and 5 are from 2003 onwards.** Before 2003 they are mostly *Accident* (18),
  *Injury*/*Injured* (9), *Excluded* (6), *Disqualified* (6) and *Engine* (5). From 2003 there is one row each for
  *Injury*, *Fuel system*, *Battery*, *Brakes* and *Engine*.
- **Many look like entries that never took the start.** From general F1 history (outside this dataset, not verified
  against it), several rows are well-known non-starts:
  - Rindt 1970, Cevert 1973, Villeneuve 1982 and Ratzenberger 1994 were fatal accidents in practice or qualifying.
  - Barrichello and Wendlinger in 1994, and Pérez at Monaco 2011, were injured in practice or qualifying.
  - Leclerc (São Paulo 2023) and Tsunoda (Monza 2021) did not make it to the start.
- **The timing differs between cases, which matters for our cutoff.** Some of these were known by the end of qualifying,
  such as injuries in practice or qualifying. Others happened after qualifying, on the way to the grid or on the
  formation lap, so they were not known at our cutoff. *Excluded* and *Disqualified* with no laps may also be
  qualifying-related decisions. *Unknown:* which rows fall into which group; the dataset has no column that says.
- **Scale.** 56 rows out of 24,761, all with a non-podium label. Any change to D3 for them would be a new decision."""),

    md("""## 7. Post-cleaning EDA

How does the cleaned table (`results_clean`, section 6) differ from raw `results`, and what does the cleaned data say
about the decisions that are still open (training window, missing-qualifying strategy, `grid = 0` as a feature)?
**Descriptive only:** nothing here changes the data or decides anything.

**Definitions used in this section**
- **Podium:** `positionOrder <= 3` (the project target). It is used here only to describe the data.
- **Non-finish:** status is neither *Finished* nor *+N Lap(s)*. *(Assumption: a car classified laps down finished the race.
  This is a reliability measure, not part of the target.)*
- **Starters per race:** cleaned rows per race. This includes the 61 *Withdrew* rows with `grid > 0` that D3 keeps by design.
- **Qualifying position:** `qualifying.position`, joined one-to-one on `(raceId, driverId)`. It is missing when there is no
  qualifying row or the position is null.
- **Era:** *before 2003* vs *2003+*. 2003 is the first season with complete qualifying data (4.3)."""),
    code("""raw_y = with_podium(with_year(results))
clean_y = with_podium(with_year(results_clean))

finish_status = tables["status"]["status"].str.fullmatch("Finished|[+][0-9]+ Laps?")
finish_ids = tables["status"].loc[finish_status, "statusId"]
clean_y["non_finish"] = ~clean_y["statusId"].isin(finish_ids)
clean_y["era"] = np.where(clean_y["year"] < 2003, "before 2003", "2003+")

clean_q = clean_y.merge(
    qualifying[["raceId", "driverId", "position", "q1"]].rename(columns={"position": "quali_position"}),
    on=["raceId", "driverId"], how="left", validate="one_to_one",
)
assert len(clean_q) == len(clean_y), "qualifying join changed the row count"

print(f"finish-like statuses: {int(finish_status.sum())} (Finished, +1 Lap, +2 Laps, ...)")
print(f"rows: raw {len(raw_y)}, cleaned {len(clean_y)}")"""),

    md("""### 7.1 Rows per season, raw vs cleaned

**Question:** in which seasons did cleaning remove rows, and how many?"""),
    code("""rows_per_season = pd.DataFrame({
    "cleaned": clean_y.groupby("year").size(),
    "raw": raw_y.groupby("year").size(),
}).fillna(0).astype(int)
rows_per_season["removed"] = rows_per_season["raw"] - rows_per_season["cleaned"]

fig, ax = plots.new_figure()
plots.style_axes(ax, "Results rows per season, raw vs cleaned", ylabel="rows")
plots.plot_lines(ax, rows_per_season[["cleaned", "raw"]])
plots.mark_splits(ax, config.TRAIN_END_YEAR, config.TEST_START_YEAR)
fig.tight_layout()
plt.show()

print("removed in total:", int(rows_per_season["removed"].sum()))
print("seasons with most rows removed:")
display(rows_per_season.sort_values("removed", ascending=False).head(6))"""),
    md("""**What it shows.** Cleaning removed 1,998 rows, and almost all of them come from before the mid-1990s.
- **Prequalifying era (D3).** The largest gaps are 1989 (620 → 418, −202), 1990 (−124) and 1991 (−123), followed by
  1981, 1978 and 1988. These are the years when many more cars entered than were allowed to start.
- **1950s (D2 and D1).** The raw line sits above the cleaned one throughout the decade. That is the Indianapolis 500
  and the shared drives.
- **Modern seasons are barely touched.** Only 9 rows were removed from 2003 onwards: none in validation (2020–21)
  and 2 in test (both D3 non-starters).

**Supports:** the impact of D1–D3. The cleaning decisions mainly change pre-1995 data, so how much they matter for the model
depends on the **training window**, which is still open. Validation and test are essentially unaffected."""),

    md("""### 7.2 Podium base rate per season, raw vs cleaned

**Question:** did cleaning change the class balance (podium rows ÷ rows), and does the base rate drift over time?"""),
    code("""podium_rate = pd.DataFrame({
    "cleaned": rate_by(clean_y, "year", "podium")["rate"],
    "raw": rate_by(raw_y, "year", "podium")["rate"],
})

fig, ax = plots.new_figure()
plots.style_axes(ax, "Podium base rate per season (podium rows / rows)", ylabel="podium rate", percent=True)
plots.plot_lines(ax, podium_rate)
plots.mark_splits(ax, config.TRAIN_END_YEAR, config.TEST_START_YEAR)
ax.set_ylim(0, None)
fig.tight_layout()
plt.show()

print(f"overall podium rate: raw {raw_y['podium'].mean():.2%}, cleaned {clean_y['podium'].mean():.2%}")
print(f"cleaned, before 2003: {clean_y.loc[clean_y['era'] == 'before 2003', 'podium'].mean():.2%} | "
      f"2003+: {clean_y.loc[clean_y['era'] == '2003+', 'podium'].mean():.2%}")
print("cleaned podium rate, lowest and highest seasons:")
display(podium_rate["cleaned"].sort_values().iloc[[0, 1, 2, -3, -2, -1]].to_frame("cleaned rate").style.format("{:.1%}"))"""),
    md("""**What it shows.**
- **Cleaning raised the base rate.** Overall it goes from 12.69% raw to 13.50% cleaned. The difference is largest in
  1988–1992, where raw drops to about 8% because non-starters inflate the number of rows. Once they are removed, the
  cleaned rate stays at about 11.5% even then.
- **The base rate follows field size.** There are 3 podiums per race, so the rate is roughly 3 ÷ the number of starters.
  - The lowest cleaned seasons are 1988–1990 (about 11.5%), which had large fields.
  - The highest are 1969 (18.6%), 1955 and 1966, which had small fields.
  - 2017–2024 sit at exactly 15% because every race had 20 cars.
- **By era:** 13.0% before 2003 and 14.3% from 2003.

**Supports:** D3, since a podium rate computed with non-starters in the denominator would be misleading. It also shows
that the class balance is about 13–15% positive, which matters later for the choice of evaluation metric. The drift
with field size is evidence for the **training window / era** decision."""),

    md("""### 7.3 Starters per race, by season (cleaned)

**Question:** how many cars line up per race, and how much does that vary within and between seasons?"""),
    code("""per_race = clean_y.groupby(["year", "raceId"]).size()
field = per_race.groupby("year").agg(["mean", "min", "max"])
x = field.index.to_numpy(dtype=float)

fig, ax = plots.new_figure()
plots.style_axes(ax, "Starters per race: season mean (line) and min-max across races (band)", ylabel="starters")
ax.fill_between(x, field["min"], field["max"], color=plots.SERIES[0], alpha=0.18, linewidth=0, zorder=2)
ax.plot(x, field["mean"], color=plots.SERIES[0], linewidth=2, zorder=3)
plots.mark_splits(ax, config.TRAIN_END_YEAR, config.TEST_START_YEAR)
ax.set_ylim(0, None)
fig.tight_layout()
plt.show()

display(field.loc[[1950, 1953, 1960, 1970, 1980, 1989, 1995, 2003, 2010, 2024]].round(1))
print("seasons with exactly 20 starters in every race:",
      [int(y) for y in field.index if field.loc[y, "min"] == field.loc[y, "max"] == 20])"""),
    md("""**What it shows.**
- **1950s–60s:** fields are small and vary a lot within a season, from 10 to 34 starters per race.
- **1970s–1995:** fields grow to 24–27 (1989: 26 in every race).
- **From 1996:** fields settle at 20–24. Every race has exactly 20 cars in 2003, 2004 and 2017–2021; 2024 has 19–20.

The counts include the 61 *Withdrew* rows with `grid > 0` that D3 keeps.

**Supports:** later feature design, plus the training window. The same grid slot means different things in different
fields: P10 of 34 is not P10 of 20. If several eras are combined, a relative position such as grid ÷ field size may be
needed *(Recommendation for the feature step, not decided here)*."""),

    md("""### 7.4 Podium rate by grid position (1–20), before 2003 vs 2003+

**Question:** how strongly does starting position predict a podium, and is that relationship the same in both eras?
`grid = 0` (meaning still pending) and grid positions above 20 are left out; their counts are printed below."""),
    code("""on_grid = clean_y[clean_y["grid"].between(1, 20)]
by_grid = rate_by(on_grid, ["grid", "era"], "podium")
grid_rate = by_grid["rate"].unstack("era")[["2003+", "before 2003"]]

fig, ax = plots.new_figure()
plots.style_axes(ax, "Podium rate by grid position", xlabel="grid position", ylabel="podium rate", percent=True)
plots.plot_lines(ax, grid_rate, marker=True)
ax.set_xticks(range(1, 21))
ax.set_ylim(0, 1)
fig.tight_layout()
plt.show()

left_out = clean_y.assign(
    grid_zero=clean_y["grid"].eq(0), grid_over_20=clean_y["grid"].gt(20)
).groupby("era")[["grid_zero", "grid_over_20"]].sum()
print("rows left out of this plot:"); display(left_out)
print("smallest n per (grid, era):", int(by_grid["n"].min()))
display(grid_rate.loc[[1, 2, 3, 4, 5, 6, 10]].style.format("{:.1%}"))"""),
    md("""**What it shows.** Grid position is strongly related to finishing on the podium in both eras, but the shape differs.
- **2003+:** the curve is much steeper. Pole converts 79.4% of the time, P2 65.0% and P3 50.9%.
- **Before 2003:** pole converts only 54.2% (P2 50.4%, P3 43.1%).
- **The curves cross at P4.** From P4 to about P10 the older era gives *more* podiums than today (P5: 27.5% vs 20.8%).
- **Rows left out:** grid 0 (51 before 2003, 77 from 2003) and grid > 20 (2,366 and 412). Each point is based on at
  least 373 rows.

**Supports:** the **training window / era** decision. A model trained across both eras would learn an average of two
different curves. It also shows why `grid = 0` needs a decided meaning before it is used as a number: it would otherwise
read as "better than pole"."""),

    md("""### 7.5 Podium rate by qualifying position (2003+)

**Question:** is qualifying position informative in the seasons where we have it, and how does it relate to grid?"""),
    code("""q2003 = clean_q[(clean_q["year"] >= 2003) & clean_q["quali_position"].between(1, 20)]
by_quali = rate_by(q2003, "quali_position", "podium")

fig, ax = plots.new_figure()
plots.style_axes(ax, "Podium rate by qualifying position, 2003+", xlabel="qualifying position",
                 ylabel="podium rate", percent=True)
plots.plot_lines(ax, by_quali[["rate"]], marker=True)
ax.set_xticks(range(1, 21))
ax.set_ylim(0, 1)
fig.tight_layout()
plt.show()

recent = clean_q[clean_q["year"] >= 2003]
same = (recent["grid"] == recent["quali_position"]).mean()
print(f"smallest n per qualifying position: {int(by_quali['n'].min())}")
print(f"2003+ rows where grid == qualifying position: {same:.1%} (differences: grid penalties, pit-lane starts, missing qualifying)")
display(by_quali.loc[[1, 2, 3, 4, 5, 6, 10], ["n", "rate"]].style.format({"rate": "{:.1%}"}))"""),
    md("""**What it shows.**
- **Qualifying position is about as informative as grid from 2003.** The podium rate is 78.5% from qualifying P1,
  62.9% from P2, 50.7% from P3, 24.3% from P5 and 4.0% from P10. Each position has at least 412 rows.
- **But grid and qualifying position are not the same thing.** They agree in only 72.4% of 2003+ rows. The others
  differ because of grid penalties, pit-lane starts and the occasional missing qualifying row.

**Supports:** the qualifying-feature questions. It is evidence that qualifying data is usable and complete from 2003,
which bears on the missing-qualifying strategy and a possible 2003 start. Whether to use grid, qualifying position or
both is a feature decision for later; their 72% agreement suggests they carry overlapping but not identical information."""),

    md("""### 7.6 Non-finish rate per decade (cleaned)

**Question:** did reliability (the share of starters that do not finish) change by era?"""),
    code("""clean_y["decade"] = (clean_y["year"] // 10 * 10).astype(int)
by_decade = rate_by(clean_y, "decade", "non_finish")

fig, ax = plots.new_figure(height=3.8)
plots.style_axes(ax, "Non-finish rate per decade (status not Finished / +N Laps)", ylabel="non-finish rate",
                 percent=True)
ax.bar([f"{d}s" for d in by_decade.index], by_decade["rate"], color=plots.SERIES[0], width=0.6, zorder=3)
ax.set_ylim(0, None)
fig.tight_layout()
plt.show()

display(by_decade.style.format({"rate": "{:.1%}"}))"""),
    md("""**What it shows.**
- **1950s–1990s:** about half of all starters did not finish (49–50%), with a peak of 56.1% in the 1980s.
- **Since then reliability has improved sharply:** 32.1% in the 2000s, 19.6% in the 2010s and 14.4% in the 2020s.

This fits 7.4. When half the field retires, a midfield starter has a real chance of a podium, and even pole is far from
safe.

**Supports:** the **training window**. Older seasons follow different race dynamics, which argues for a later start
date, an era feature, or at least era-split evaluation. *(Evidence only; the window is not decided here.)*"""),

    md("""### 7.7 Missingness per season: grid, qualifying position, q1 (cleaned)

**Question:** in which seasons are the pre-race fields missing? `grid` is never null (asserted below), so instead of a flat line
at 0% the chart shows the share of `grid = 0`. `q1 missing` is drawn dashed on top of `qualifying position missing`,
because the two lines overlap for most seasons. That value is **not** missing, but its meaning differs by era (4.1), so it is worth seeing alongside."""),
    code("""assert clean_q["grid"].notna().all(), "grid has missing values"
print("grid missing: 0 rows in every season (asserted), so it is not drawn as a line.")

missing_by_season = clean_q.assign(
    quali_missing=clean_q["quali_position"].isna(),
    q1_missing=clean_q["q1"].isna(),
    grid_zero=clean_q["grid"].eq(0),
).groupby("year")[["quali_missing", "q1_missing", "grid_zero"]].mean()
missing_by_season.columns = [
    "qualifying position missing", "q1 missing", "grid = 0 (not missing; meaning pending)",
]

fig, ax = plots.new_figure()
plots.style_axes(ax, "Share of cleaned rows per season: missing qualifying fields and grid = 0",
                 ylabel="share of rows", percent=True)
# q1 is drawn dashed over a thicker solid line so the two overlapping series stay visible.
plots.plot_lines(ax, missing_by_season, linestyles=["-", "--", "-"], linewidths=[3.5, 2, 2])
plots.mark_splits(ax, config.TRAIN_END_YEAR, config.TEST_START_YEAR)
ax.set_ylim(0, 1.05)
fig.tight_layout()
plt.show()

eras = clean_q.assign(
    quali_missing=clean_q["quali_position"].isna(), q1_missing=clean_q["q1"].isna(), grid_zero=clean_q["grid"].eq(0)
)
display(eras.groupby("era")[["quali_missing", "q1_missing", "grid_zero"]].mean().style.format("{:.1%}"))
print("2003+ seasons with the most q1 missing:")
display(missing_by_season.loc[2003:, ["qualifying position missing", "q1 missing"]]
        .sort_values("q1 missing", ascending=False).head(4).style.format("{:.1%}"))"""),
    md("""**What it shows.**
- **`grid` is never missing** (asserted).
- **Qualifying position** is missing for every row before 1994, for 53.7% of rows in 1994–2002, and for only 0.26% of
  rows from 2003. Before 2003 overall it is 90.2%.
- **`q1`** follows the same pattern but is a little less complete from 2003: 1.64% missing, worst in 2004 (5.8%),
  2005 (4.5%) and 2003 (3.4%). Those seasons used different qualifying formats. Why the time is missing while a
  position exists is *Unknown*.
- **`grid = 0`** is rare after cleaning: 0.32% of rows before 2003 and 0.86% from 2003. It drifts upward in the 2020s,
  in line with more pit-lane starts.

**Supports:**
- The **missing-qualifying strategy and the training window:** from 2003 the qualifying fields are essentially complete,
  while before 2003 they are mostly absent.
- **Feature choice:** qualifying position is more complete than the `q1` time.
- **The meaning of `grid = 0`:** it is a small but growing modern share.

*(All evidence only; none of these decisions is made here.)*"""),
]

nb = new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3 (ipykernel)", "language": "python"}
nb.metadata["language_info"] = {"name": "python"}
nbformat.validate(nb)
out.parent.mkdir(parents=True, exist_ok=True)
nbformat.write(nb, out)
print("wrote", out)
