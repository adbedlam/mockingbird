from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass
class Message:
    author: str
    timestamp: datetime
    text: str | None = None
    picture_description: str | None = None
    picture_path: Path | None = None
    message_id: str | None = None

    def to_sample(self) -> str:
        timestamp = self.timestamp.isoformat()
        text = f" | {self.text}" if self.text else ""
        picture = (
            f" | picture_description: {self.picture_description}"
            if self.picture_description
            else ""
        )

        return f"{self.author}@{timestamp} | {text}{picture}"
