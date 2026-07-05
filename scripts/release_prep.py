"""Release preparation: version bump + changelog regeneration.

Invoked as ``just release-prep X.Y.Z`` (which also runs the absolute perf
gates via ``just bench`` afterwards). This script:

1. Validates the requested version (SemVer, optional ``rcN`` suffix).
2. Sets ``[project] version`` in ``pyproject.toml`` (the single version
   source; the Rust crate version never ships).
3. Regenerates ``CHANGELOG.md`` with git-cliff as of the would-be tag.
4. Verifies the new changelog actually contains the release section, so the
   ``release-guard`` CI job cannot be surprised later.

Usage:
    uv run python scripts/release_prep.py 0.1.0
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path
from typing import NoReturn

REPO_ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = REPO_ROOT / "pyproject.toml"
CHANGELOG = REPO_ROOT / "CHANGELOG.md"
VERSION_RE = re.compile(r"^\d+\.\d+\.\d+(rc\d+)?$")


def fail(msg: str) -> NoReturn:
    """Print an error and exit non-zero."""
    print(f"release-prep: error: {msg}", file=sys.stderr)
    raise SystemExit(1)


def set_pyproject_version(version: str) -> None:
    text = PYPROJECT.read_text(encoding="utf-8")
    new_text, n = re.subn(
        r'(?m)^version = "[^"]+"$', f'version = "{version}"', text, count=1
    )
    if n != 1:
        fail("could not find the [project] version line in pyproject.toml")
    if new_text != text:
        PYPROJECT.write_text(new_text, encoding="utf-8")
        print(f"release-prep: pyproject.toml version -> {version}")
    else:
        print(f"release-prep: pyproject.toml already at {version}")


def regenerate_changelog(version: str) -> None:
    cmd = ["uv", "run", "git-cliff", "--tag", f"v{version}", "-o", str(CHANGELOG)]
    result = subprocess.run(cmd, cwd=REPO_ROOT, check=False)
    if result.returncode != 0:
        fail("git-cliff failed; is the dev group synced? (uv sync --all-groups)")
    if f"## [{version}]" not in CHANGELOG.read_text(encoding="utf-8"):
        fail(f"CHANGELOG.md has no section for {version} after regeneration")
    print(f"release-prep: CHANGELOG.md regenerated with a {version} section")


def main() -> None:
    if len(sys.argv) != 2:
        fail("usage: release_prep.py X.Y.Z[rcN]")
    version = sys.argv[1].removeprefix("v")
    if not VERSION_RE.match(version):
        fail(f"{version!r} is not X.Y.Z or X.Y.ZrcN")
    set_pyproject_version(version)
    regenerate_changelog(version)
    print(
        "release-prep: done — review the diff, open a PR titled "
        f"'release: v{version}', and after merge push an annotated tag "
        f"v{version} on the merge commit."
    )


if __name__ == "__main__":
    main()
