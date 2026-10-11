# CLAUDE.md — F1 Podium Prediction (CSEN903)

## Project
University team project, multi-milestone. Milestone 1 due 18 Oct 2026.
Predict, for each driver in each race, whether they finish on the podium.

- **Dataset:** Kaggle "Formula 1 World Championship (1950–2024)" (Ergast), 14 CSVs in `data/raw/`.
- **Grain:** one row per `(raceId, driverId)`. Must be unique — assert it.
- **Target:** `podium = results.positionOrder <= 3`.
- **Prediction cutoff:** after qualifying, before the race starts.
- **Split (planned):** train ≤ 2019, validation 2020–2021, test ≥ 2022 (by `races.year`).
- **Deliverables:** one run-all Jupyter notebook (must also run on Kaggle), GitHub repo, PDF report.

## Leakage rules (most important)
Ask for every feature: "Could we have known this before lights out?"

- Allowed for the current race: `results.grid`, `qualifying.*`, static metadata (races, circuits, drivers, constructors).
- Forbidden for the current race: `results` position/positionText/points/laps/time/milliseconds/fastestLap*/statusId, `status`, `lap_times`, `pit_stops`, `constructor_results`, and standings rows for the same race.
- Standings tables are cumulative AFTER each race → use only the previous race's standings.
- Form/rolling features must use only earlier races: sort chronologically by `(year, round)` or `date`, NEVER by `raceId`, and `shift(1)` before rolling.
- Fit imputers/scalers/encoders on training data only.
- Never tune hyperparameters or thresholds on the test set.
- Write programmatic checks for these rules; never rely on eyeballing.

## Data rules
- `data/raw/` is immutable. Clean into new DataFrames/files only.
- `\N` is the missing-value sentinel.
- After every join: check row counts, key uniqueness, unmatched foreign keys, unexpected nulls.
- Do not drop duplicates casually. `(raceId, driverId)` duplicates are mostly 1950s shared drives — investigate and apply a documented rule.
- Open decisions (do not decide silently — propose and wait for approval): shared-drive rule, Indy 500 exclusion, non-starters (DNQ/DNPQ), grid = 0 meaning, training window, missing-qualifying strategy, sprint data, standings at season start.

## Repo conventions
- Reusable code in `src/f1podium/` (loading, validation, features, evaluation); analysis and narrative in `notebooks/`.
- `scripts/build_notebook.py` is the source of truth for `notebooks/milestone1.ipynb`. Never hand-edit the `.ipynb`: change the generator, rebuild (`python scripts/build_notebook.py`), then re-execute with `jupyter nbconvert --to notebook --execute --inplace notebooks/milestone1.ipynb` using the venv (its `Scripts` folder first on PATH, or the kernel picks up the global Python).
- Config (paths, split years, random seed) centralised; no hardcoded absolute paths.
- Small, meaningful commits with descriptive messages. Never commit data, venvs, caches, or secrets.

## How to work with me (Youssef)
- I am learning: before running a command or making a non-trivial change, explain what it does and why.
- Inspect before changing. Propose a plan for substantial changes and wait for my approval.
- Make changes in small steps; run checks after each; show what changed.
- When a check or assertion fails, investigate the cause — never delete the check to make it pass.
- Label choices as Requirement / Recommendation / Assumption / Experiment / Unknown.
- I am a beginner with Bash/terminal and Python environments — explain those commands plainly.
