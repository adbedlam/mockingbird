from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TerminalOptions:
    state_dir: Path | None
    seed: int | None
    refresh_persona: bool
    no_refresh_persona: bool
    fast_replies: bool
    max_generations: int | None