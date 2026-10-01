"""Keep wren's version in sync across VERSION, pyproject.toml, and __init__.py.

    python scripts/version.py          # print the current version
    python scripts/version.py 0.3.0    # bump to 0.3.0 across all sources

`VERSION` (repo root) is the source of truth. Mirrors ymlx's convention of a
single version file kept in sync with the package metadata.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VERSION_FILE = ROOT / "VERSION"
PYPROJECT = ROOT / "pyproject.toml"
INIT = ROOT / "src" / "wren" / "__init__.py"


def current() -> str:
    return VERSION_FILE.read_text().strip()


def _set(ver: str) -> None:
    text = PYPROJECT.read_text()
    new = re.sub(r'(^version\s*=\s*)"[^"]*"(\s*$)', rf'\g<1>"{ver}"\g<2>', text, flags=re.M)
    if new == text:
        raise SystemExit(f"version.py: could not find 'version = ...' in {PYPROJECT}")
    PYPROJECT.write_text(new)

    # src/wren/__init__.py __version__
    text = INIT.read_text()
    new = re.sub(r'(__version__\s*=\s*)"[^"]*"', rf'\g<1>"{ver}"', text)
    if new == text:
        raise SystemExit(f"version.py: could not find '__version__ = ...' in {INIT}")
    INIT.write_text(new)

    VERSION_FILE.write_text(ver + "\n")


def main(argv) -> int:
    if len(argv) == 0:
        print(current())
        return 0
    if len(argv) == 1:
        ver = argv[0].strip()
        if not re.fullmatch(r"\d+\.\d+\.\d+", ver):
            raise SystemExit(f"version.py: '{ver}' is not a valid semver (X.Y.Z).")
        _set(ver)
        print(f"bumped to {ver} (VERSION, pyproject.toml, __init__.py)")
        return 0
    raise SystemExit("version.py: usage: python scripts/version.py [X.Y.Z]")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
