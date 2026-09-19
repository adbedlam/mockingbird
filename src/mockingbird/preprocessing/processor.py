from pathlib import Path
from typing import Protocol

from mockingbird.datacls import Message


class Processor(Protocol):
    def process(self, data_path: Path, *args, **kwargs) -> list[Message]: ...
