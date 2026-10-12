"""Tests for the notebook Setup cell's Kaggle src lookup, using temp directory trees.

The function lives in a code-cell string inside scripts/build_notebook.py, so we pull that
source out with ast (without running the script, which would write a notebook) and exec it.
"""

import ast
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_notebook.py"


@pytest.fixture(scope="module")
def setup_ns():
    for node in ast.walk(ast.parse(SCRIPT.read_text(encoding="utf-8"))):
        if (isinstance(node, ast.Constant) and isinstance(node.value, str)
                and "def _add_src_to_path" in node.value):
            source = node.value.split("SRC_DIR, SRC_COMMIT =")[0]
            break
    else:
        pytest.fail("Setup cell not found in build_notebook.py")
    ns = {}
    exec(source, ns)
    return ns


@pytest.fixture
def restore_path():
    saved = list(sys.path)
    yield
    sys.path[:] = saved


def make_src(src: Path, version_in_src: bool, version="abc123"):
    (src / "f1podium").mkdir(parents=True)
    (src / "f1podium" / "__init__.py").write_text("")
    version_file = src / "VERSION" if version_in_src else src.parent / "VERSION"
    version_file.write_text(version + "\n")


@pytest.mark.parametrize("version_in_src", [True, False])
def test_finds_nested_src_with_version_in_either_place(setup_ns, restore_path, tmp_path, version_in_src):
    src = tmp_path / "datasets" / "someone" / "f1podium-src" / "versions" / "src"  # depth 5
    make_src(src, version_in_src)
    found, commit = setup_ns["_add_src_to_path"](tmp_path)
    assert found == src and commit == "abc123"
    assert sys.path[0] == str(src)


def test_ignores_src_without_f1podium_and_goes_on_to_next(setup_ns, restore_path, tmp_path):
    (tmp_path / "a" / "src").mkdir(parents=True)  # a src/ with no f1podium/
    (tmp_path / "a" / "VERSION").write_text("wrong\n")
    make_src(tmp_path / "b" / "src", version_in_src=True, version="right")
    found, commit = setup_ns["_add_src_to_path"](tmp_path)
    assert found == tmp_path / "b" / "src" and commit == "right"


def test_depth_limit(setup_ns, restore_path, tmp_path):
    make_src(tmp_path / "1" / "2" / "3" / "4" / "5" / "6" / "7" / "src", version_in_src=True)  # depth 8
    with pytest.raises(FileNotFoundError):
        setup_ns["_add_src_to_path"](tmp_path)


def test_no_match_raises_with_tree_and_does_not_fall_back_to_repo(setup_ns, restore_path, tmp_path):
    (tmp_path / "my-dataset" / "nested").mkdir(parents=True)
    (tmp_path / "my-dataset" / "results.csv").write_text("")
    before = list(sys.path)
    with pytest.raises(FileNotFoundError) as exc:
        setup_ns["_add_src_to_path"](tmp_path)  # cwd is the repo, which has src/f1podium
    assert "my-dataset" in str(exc.value) and "results.csv" in str(exc.value)
    assert sys.path == before


def test_error_tree_is_limited_to_depth_4(setup_ns, tmp_path):
    (tmp_path / "d1" / "d2" / "d3" / "d4" / "d5").mkdir(parents=True)
    tree = setup_ns["_describe_tree"](tmp_path)
    assert "d4/" in tree and "d5/" not in tree


def test_uses_repo_src_when_kaggle_input_missing(setup_ns, restore_path, tmp_path):
    src, commit = setup_ns["_add_src_to_path"](tmp_path / "does-not-exist")
    assert (src / "f1podium").is_dir() and commit is None


@pytest.mark.parametrize("parts", [("f1podium-src",), ("datasets", "owner", "f1podium-src")])
def test_flat_and_datasets_owner_slug_layouts(setup_ns, restore_path, tmp_path, parts):
    src = tmp_path.joinpath(*parts) / "src"
    make_src(src, version_in_src=True)
    found, commit = setup_ns["_add_src_to_path"](tmp_path)
    assert found == src and commit == "abc123"


def _setup_cell_source():
    for node in ast.walk(ast.parse(SCRIPT.read_text(encoding="utf-8"))):
        if (isinstance(node, ast.Constant) and isinstance(node.value, str)
                and "def _add_src_to_path" in node.value):
            return node.value
    pytest.fail("Setup cell not found in build_notebook.py")


def test_autoreload_is_guarded_and_never_a_bare_magic():
    src = _setup_cell_source()
    lines = [line.strip() for line in src.splitlines()]
    assert not any(line.startswith(("%load_ext", "%autoreload")) for line in lines)
    assert 'run_line_magic("load_ext", "autoreload")' in src
    try_at = src.index("try:\n        get_ipython()")
    assert try_at < src.index('run_line_magic("load_ext"') < src.index("except Exception")
    assert 'print(f"autoreload unavailable: {exc}")' in src
    assert 'Path("/kaggle/input").exists()' in src
