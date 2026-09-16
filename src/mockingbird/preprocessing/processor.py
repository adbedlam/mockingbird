from abc import ABC, abstractmethod
from pathlib import Path

from src.mockingbird.datacls import Message


class Processor(ABC):
    @abstractmethod
    def process(self, data_path: Path, *args, **kwargs) -> list[Message]:
        pass
