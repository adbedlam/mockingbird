from pathlib import Path
from typing import Protocol

from mockingbird.conversation.history import History


class Processor(Protocol):
    def process(
        self, user: str, other: str, data_path: Path, *args, **kwargs
    ) -> History: ...
