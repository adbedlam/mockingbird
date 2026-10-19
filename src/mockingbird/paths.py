from pathlib import Path


def _find_project_root(start: Path) -> Path:
    for p in [start, *start.parents]:
        if (p / "pyproject.toml").exists():
            return p
    raise RuntimeError(f"no pyproject.toml found above {start}")


PROJECT_ROOT = _find_project_root(Path(__file__).resolve())
