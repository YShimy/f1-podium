"""Package src/ for Kaggle: dist/f1podium_src_<shorthash>.zip.

The zip contains src/ (the files git tracks there) and nothing else, plus src/VERSION
holding the full commit hash. Upload it as a Kaggle dataset; the notebook finds
<dataset>/src/VERSION, puts <dataset>/src on sys.path and prints the hash.

Refuses to run if the git working tree is dirty, so the hash always identifies exactly
the code in the zip.

Usage (from anywhere, with any Python):
    python scripts/package_kaggle.py
"""

import subprocess
import sys
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)  # fixed, so the same commit always gives the same zip bytes


def git(repo_root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo_root), *args], capture_output=True, text=True, check=True
    )
    return result.stdout


def ensure_clean(repo_root: Path) -> None:
    """Exit with an error if there are uncommitted or untracked (non-ignored) changes."""
    dirty = git(repo_root, "status", "--porcelain").strip()
    if dirty:
        raise SystemExit(
            "Refusing to package: the git working tree is dirty. Commit or stash first.\n" + dirty
        )


def package(repo_root: Path = REPO_ROOT) -> tuple[Path, str]:
    """Build the zip for the current commit and return (zip path, full commit hash)."""
    ensure_clean(repo_root)
    full_hash = git(repo_root, "rev-parse", "HEAD").strip()
    short_hash = git(repo_root, "rev-parse", "--short", "HEAD").strip()

    # Tracked files only: no __pycache__, no local junk. The tree is clean, so these equal the working copy.
    tracked = [p for p in git(repo_root, "ls-files", "src").splitlines() if p and p != "src/VERSION"]
    if not tracked:
        raise SystemExit("Nothing to package: no tracked files under src/.")

    out_dir = repo_root / "dist"
    out_dir.mkdir(exist_ok=True)
    zip_path = out_dir / f"f1podium_src_{short_hash}.zip"

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for rel in sorted(tracked):
            info = zipfile.ZipInfo(rel, date_time=ZIP_TIMESTAMP)
            info.compress_type = zipfile.ZIP_DEFLATED
            zf.writestr(info, (repo_root / rel).read_bytes())
        version = zipfile.ZipInfo("src/VERSION", date_time=ZIP_TIMESTAMP)
        zf.writestr(version, full_hash + "\n")

    with zipfile.ZipFile(zip_path) as zf:  # verify: only src/ entries, VERSION matches
        names = zf.namelist()
        assert all(n.startswith("src/") for n in names), "zip contains something outside src/"
        assert zf.read("src/VERSION").decode().strip() == full_hash
    return zip_path, full_hash


def main() -> int:
    zip_path, full_hash = package()
    print(f"wrote   {zip_path}")
    print(f"commit  {full_hash}")
    print("Upload it as a new version of the Kaggle dataset, then check that the notebook prints this commit.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
