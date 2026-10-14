"""Tests for the EDA helpers on a small fake table."""

import pandas as pd
import pytest

from f1podium.eda import rate_by, with_podium


@pytest.fixture
def rows():
    return pd.DataFrame({
        "group": ["a", "a", "a", "b", "b"],
        "positionOrder": pd.array([1, 3, 4, 2, None], dtype="Int64"),
    })


def test_with_podium_marks_top_three_and_treats_missing_as_false(rows):
    out = with_podium(rows)
    assert out["podium"].tolist() == [True, True, False, True, False]
    assert out["podium"].dtype == bool
    assert "podium" not in rows.columns  # input not modified


def test_rate_by_counts_and_rates(rows):
    out = rate_by(with_podium(rows), "group", "podium")
    assert out.loc["a", "n"] == 3 and out.loc["a", "n_flag"] == 2
    assert out.loc["b", "n"] == 2 and out.loc["b", "n_flag"] == 1
    assert out.loc["a", "rate"] == pytest.approx(2 / 3)
    assert out.loc["b", "rate"] == pytest.approx(0.5)


def test_rate_by_rejects_missing_flags(rows):
    rows["flag"] = pd.array([True, None, False, True, False], dtype="boolean")
    with pytest.raises(ValueError, match="missing"):
        rate_by(rows, "group", "flag")


# --- feature evidence helpers (section 9.6-9.7) ----------------------------------------------------------

def test_rate_by_bins_with_edges_and_a_missing_bucket():
    from f1podium.eda import rate_by_bins
    x = pd.Series([1, 2, 3, 5, 12, None], name="grid")
    y = [1, 1, 0, 1, 0, 1]
    out = rate_by_bins(x, y, bins=[0, 1, 3, 10, 99], labels=["1", "2-3", "4-10", "11+"])
    assert list(out.index) == ["1", "2-3", "4-10", "11+", "missing"]
    assert out["n"].tolist() == [1, 2, 1, 1, 1] and out["podiums"].tolist() == [1, 1, 1, 0, 1]
    assert out.loc["2-3", "rate"] == 0.5 and out.index.name == "grid"


def test_rate_by_bins_quantiles_drop_duplicate_edges():
    from f1podium.eda import rate_by_bins
    x = pd.Series([0.0] * 8 + [0.5, 1.0])                   # mostly zeros: deciles collapse
    out = rate_by_bins(x, [0] * 8 + [1, 1], q=10)
    assert out["n"].sum() == 10 and len(out) < 10 and "missing" not in out.index
    with pytest.raises(ValueError):
        rate_by_bins(x, [0] * 10)


def test_univariate_pr_auc_finds_the_direction_and_ignores_missing_rows():
    from sklearn.metrics import average_precision_score
    from f1podium.eda import univariate_pr_auc
    y = pd.Series([1, 1, 0, 0, 0, 0, 1, 0])
    X = pd.DataFrame({"grid": [1, 2, 8, 9, 10, 7, 3, 6],              # lower is better
                      "points": [50, 40, 1, 2, 0, 3, 30, None]})      # higher is better, one missing
    out = univariate_pr_auc(X, y)
    assert out.loc["grid", "direction"] == "lower" and out.loc["points", "direction"] == "higher"
    assert out.loc["grid", "pr_auc"] == pytest.approx(average_precision_score(y, -X["grid"]))
    assert out.loc["points", "n"] == 7 and out.loc["points", "coverage"] == pytest.approx(7 / 8)
    assert out.loc["points", "base_rate"] == pytest.approx(3 / 7)
    assert out.loc["grid", "lift"] == pytest.approx(out.loc["grid", "pr_auc"] / 0.375)
