from pathlib import Path
from typing import Protocol

from mockingbird.conversation.history import History


class Processor(Protocol):
    def process(
        self,
        user: str,
        other: str,
        data_dir: Path,
        start: int = 0,
        end: int | None = None,
    ) -> History: ...
