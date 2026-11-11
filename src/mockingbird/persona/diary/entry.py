from dataclasses import dataclass
from datetime import datetime


@dataclass
class Entry:
    timestamp: datetime
    content: str

    def to_string(self) -> str:
        return self.timestamp.isoformat() + " | " + self.content