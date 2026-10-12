"""Tests for scripts/package_kaggle.py, using a throwaway git repository."""

import importlib.util
import shutil
import subprocess
import zipfile
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git is not available")

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "package_kaggle.py"


@pytest.fixture(scope="module")
def pk():
    spec = importlib.util.spec_from_file_location("package_kaggle", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_git(repo: Path, *args: str) -> str:
    cmd = ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@example.com",
           "-c", "commit.gpgsign=false", *args]
    return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()


@pytest.fixture
def repo(tmp_path):
    run_git(tmp_path, "init", "-q")
    (tmp_path / "src" / "f1podium").mkdir(parents=True)
    (tmp_path / "src" / "f1podium" / "__init__.py").write_text("")
    (tmp_path / "src" / "f1podium" / "load.py").write_text("X = 1")
    (tmp_path / "notebooks").mkdir()
    (tmp_path / "notebooks" / "n.ipynb").write_text("{}")
    (tmp_path / ".gitignore").write_text("dist/\n__pycache__/\n")
    run_git(tmp_path, "add", "-A")
    run_git(tmp_path, "commit", "-q", "-m", "init")
    return tmp_path


def test_zip_contains_only_src_and_version_with_full_hash(pk, repo):
    (repo / "src" / "f1podium" / "__pycache__").mkdir()
    (repo / "src" / "f1podium" / "__pycache__" / "x.pyc").write_bytes(b"junk")  # ignored, untracked

    zip_path, full_hash = pk.package(repo)

    short = run_git(repo, "rev-parse", "--short", "HEAD")
    assert full_hash == run_git(repo, "rev-parse", "HEAD") and len(full_hash) == 40
    assert zip_path == repo / "dist" / f"f1podium_src_{short}.zip"
    with zipfile.ZipFile(zip_path) as zf:
        assert sorted(zf.namelist()) == ["src/VERSION", "src/f1podium/__init__.py", "src/f1podium/load.py"]
        assert zf.read("src/VERSION").decode() == full_hash + "\n"


def test_refuses_when_working_tree_is_dirty(pk, repo):
    (repo / "src" / "f1podium" / "load.py").write_text("X = 2")  # modified, uncommitted
    with pytest.raises(SystemExit, match="dirty"):
        pk.package(repo)
    assert not (repo / "dist").exists()


def test_refuses_on_untracked_file(pk, repo):
    (repo / "src" / "f1podium" / "new.py").write_text("Y = 1")
    with pytest.raises(SystemExit, match="dirty"):
        pk.package(repo)


def test_same_commit_gives_identical_zip(pk, repo):
    first, _ = pk.package(repo)
    data = first.read_bytes()
    second, _ = pk.package(repo)
    assert second.read_bytes() == data
