from pathlib import Path

import yaml

_EXAMPLE = Path(__file__).resolve().parents[1] / "config.example.yaml"
_SCHEDULER_DEFAULTS = yaml.safe_load(_EXAMPLE.read_text(encoding="utf-8"))["scheduler"]


def scheduler_settings(**overrides: int | float | str) -> dict:
    return {**_SCHEDULER_DEFAULTS, **overrides}
