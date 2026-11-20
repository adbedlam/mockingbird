from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass(frozen=True)
class TelegramOptions:
    state_dir: Path | None
    command: Literal[
        "status",
        "observe",
        "export",
        "prepare-test",
        "send-test",
        "confirm-appearance",
        "resolve-delivery",
        "run",
    ]
    live: bool
    seconds: int
    recipient: int | None
    text: str | None
    approval_id: str | None
    confirm_text: str | None
    message_id: int | None
    request_id: str | None
    outcome: Literal["sent", "not-sent"] | None
    fast_replies: bool
    max_generations: int | None
