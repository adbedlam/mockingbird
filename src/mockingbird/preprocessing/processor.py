from pathlib import Path
from typing import Protocol

from mockingbird.conversation.history import History


class Processor(Protocol):
    def process(
        self, data_path: Path, user: str, other: str, *args, **kwargs
    ) -> History: ...
