"""Reusable data-integrity checks.

Check functions are pure: they return results (dicts / DataFrames) and never
print or assert. The raw audit reports them; later pipeline steps can turn
them into hard failures with `assert_all_passed`.
"""

import pandas as pd

# (table, key columns). Surrogate ids first, then natural/composite keys.
PRIMARY_KEYS: list[tuple[str, list[str]]] = [
    ("circuits", ["circuitId"]),
    ("constructors", ["constructorId"]),
    ("drivers", ["driverId"]),
    ("races", ["raceId"]),
    ("status", ["statusId"]),
    ("seasons", ["year"]),
    ("results", ["resultId"]),
    ("sprint_results", ["resultId"]),
    ("qualifying", ["qualifyId"]),
    ("driver_standings", ["driverStandingsId"]),
    ("constructor_standings", ["constructorStandingsId"]),
    ("constructor_results", ["constructorResultsId"]),
    ("circuits", ["circuitRef"]),
    ("constructors", ["constructorRef"]),
    ("drivers", ["driverRef"]),
    ("races", ["year", "round"]),
    ("lap_times", ["raceId", "driverId", "lap"]),
    ("pit_stops", ["raceId", "driverId", "stop"]),
    ("driver_standings", ["raceId", "driverId"]),
    ("constructor_standings", ["raceId", "constructorId"]),
    ("constructor_results", ["raceId", "constructorId"]),
    ("qualifying", ["raceId", "driverId"]),
    ("results", ["raceId", "driverId"]),  # expected to fail: shared drives
    ("sprint_results", ["raceId", "driverId"]),
]

# (child table, fk column, parent table, parent key column)
FOREIGN_KEYS: list[tuple[str, str, str, str]] = [
    ("races", "circuitId", "circuits", "circuitId"),
    ("races", "year", "seasons", "year"),
    *[
        (child, col, parent, col)
        for child in ("results", "sprint_results", "qualifying")
        for col, parent in (
            ("raceId", "races"),
            ("driverId", "drivers"),
            ("constructorId", "constructors"),
        )
    ],
    ("results", "statusId", "status", "statusId"),
    ("sprint_results", "statusId", "status", "statusId"),
    ("driver_standings", "raceId", "races", "raceId"),
    ("driver_standings", "driverId", "drivers", "driverId"),
    ("constructor_standings", "raceId", "races", "raceId"),
    ("constructor_standings", "constructorId", "constructors", "constructorId"),
    ("constructor_results", "raceId", "races", "raceId"),
    ("constructor_results", "constructorId", "constructors", "constructorId"),
    ("lap_times", "raceId", "races", "raceId"),
    ("lap_times", "driverId", "drivers", "driverId"),
    ("pit_stops", "raceId", "races", "raceId"),
    ("pit_stops", "driverId", "drivers", "driverId"),
]


def check_primary_key(df: pd.DataFrame, key_cols: list[str], table: str) -> dict:
    """A valid key has no nulls and no repeated values."""
    n_null = int(df[key_cols].isna().any(axis=1).sum())
    n_dup = int(df.duplicated(subset=key_cols, keep=False).sum())
    return {
        "table": table,
        "key": ", ".join(key_cols),
        "n_rows": len(df),
        "n_null_key": n_null,
        "n_duplicate_rows": n_dup,
        "passed": n_null == 0 and n_dup == 0,
    }


def check_foreign_key(
    child: pd.DataFrame,
    child_name: str,
    fk_col: str,
    parent: pd.DataFrame,
    parent_name: str,
    pk_col: str,
) -> dict:
    """Every non-null child value must exist in the parent key column."""
    fk = child[fk_col]
    non_null = fk.dropna()
    unmatched = non_null[~non_null.isin(parent[pk_col].dropna())]
    n_null = int(fk.isna().sum())
    return {
        "child": child_name,
        "fk": fk_col,
        "parent": f"{parent_name}.{pk_col}",
        "n_child_rows": len(child),
        "n_null_fk": n_null,
        "n_unmatched_rows": len(unmatched),
        "n_unmatched_distinct": int(unmatched.nunique()),
        "example_unmatched": sorted(unmatched.unique().tolist())[:5],
        "passed": n_null == 0 and len(unmatched) == 0,
    }


def find_duplicates(df: pd.DataFrame, key_cols: list[str]) -> pd.DataFrame:
    """All rows whose key appears more than once (for inspection, not dropping)."""
    return df[df.duplicated(subset=key_cols, keep=False)].sort_values(key_cols)


def missing_summary(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """One row per (table, column): missing count and percentage."""
    rows = []
    for name, df in tables.items():
        n = len(df)
        for col in df.columns:
            n_missing = int(df[col].isna().sum())
            rows.append({
                "table": name,
                "column": col,
                "dtype": str(df[col].dtype),
                "n_rows": n,
                "n_missing": n_missing,
                "pct_missing": round(100 * n_missing / n, 2) if n else 0.0,
            })
    return pd.DataFrame(rows)


def run_primary_key_checks(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    return pd.DataFrame(
        [check_primary_key(tables[t], cols, t) for t, cols in PRIMARY_KEYS]
    )


def run_foreign_key_checks(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    return pd.DataFrame([
        check_foreign_key(tables[c], c, fk, tables[p], p, pk)
        for c, fk, p, pk in FOREIGN_KEYS
    ])


def assert_all_passed(results: list[dict] | pd.DataFrame) -> None:
    """Raise if any check failed, listing every failure."""
    df = pd.DataFrame(results)
    failed = df[~df["passed"]]
    if len(failed):
        raise AssertionError(
            f"{len(failed)} check(s) failed:\n{failed.to_string(index=False)}"
        )
