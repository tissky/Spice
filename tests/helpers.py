from __future__ import annotations

from pathlib import Path


def repo_root(start: Path | None = None) -> Path:
    """Return the repository root regardless of the calling test's depth."""

    current = (start or Path(__file__)).resolve()
    for parent in (current, *current.parents):
        if (parent / "pyproject.toml").exists() and (parent / "spice").is_dir():
            return parent
    raise RuntimeError(f"Could not find Spice repository root from {current}")
