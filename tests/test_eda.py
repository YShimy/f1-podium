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
