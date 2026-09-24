from pathlib import Path
from typing import Protocol

from mockingbird.datacls import Conversation


class Processor(Protocol):
    def process(self, data_path: Path, *args, **kwargs) -> Conversation: ...
