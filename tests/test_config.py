"""Tests for data-directory discovery in config.find_raw_dir."""

import pytest

from f1podium.config import PROJECT_ROOT, find_raw_dir


@pytest.fixture(autouse=True)
def no_env_override(monkeypatch):
    monkeypatch.delenv("F1_DATA_DIR", raising=False)


def test_uses_folder_containing_races_csv_under_kaggle_input(tmp_path):
    data = tmp_path / "input" / "formula-1-world-championship" / "nested"
    data.mkdir(parents=True)
    (data / "races.csv").write_text("raceId,year")  # content is irrelevant; only the file name is searched
    (tmp_path / "input" / "other-dataset").mkdir()
    assert find_raw_dir(kaggle_input=tmp_path / "input") == data


def test_flat_layout(tmp_path):
    data = tmp_path / "input" / "f1-data"
    data.mkdir(parents=True)
    (data / "races.csv").write_text("")
    assert find_raw_dir(kaggle_input=tmp_path / "input") == data


def test_new_datasets_owner_slug_layout(tmp_path):
    data = tmp_path / "input" / "datasets" / "rohanrao" / "formula-1-world-championship-1950-2020"
    data.mkdir(parents=True)
    (data / "races.csv").write_text("")
    assert find_raw_dir(kaggle_input=tmp_path / "input") == data


def test_depth_limit(tmp_path):
    data = tmp_path / "input" / "1" / "2" / "3" / "4" / "5" / "6" / "7"
    data.mkdir(parents=True)
    (data / "races.csv").write_text("")
    with pytest.raises(FileNotFoundError):
        find_raw_dir(kaggle_input=tmp_path / "input")


def test_raises_with_tree_listing_when_no_races_csv(tmp_path):
    (tmp_path / "input" / "datasets" / "someone" / "other-dataset").mkdir(parents=True)
    (tmp_path / "input" / "datasets" / "someone" / "other-dataset" / "drivers.csv").write_text("")
    with pytest.raises(FileNotFoundError, match="races.csv") as exc:
        find_raw_dir(kaggle_input=tmp_path / "input")
    assert "other-dataset" in str(exc.value) and "drivers.csv" in str(exc.value)


def test_falls_back_to_repo_data_raw_when_no_kaggle_input(tmp_path):
    assert find_raw_dir(kaggle_input=tmp_path / "missing") == PROJECT_ROOT / "data" / "raw"


def test_env_var_overrides_everything(tmp_path, monkeypatch):
    monkeypatch.setenv("F1_DATA_DIR", str(tmp_path / "custom"))
    assert find_raw_dir(kaggle_input=tmp_path / "input") == tmp_path / "custom"
