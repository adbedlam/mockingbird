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
    message_id: str | int | None = None

    def to_sample(self):
        author_timestamp = f"{self.author}@{self.timestamp}"
        text = f" | {self.text}" if self.text else ""
        picture = ""
        if self.picture_description:
            picture = f" | picture_description: {self.picture_description}"
        elif self.picture_path:
            picture = " | picture attached"
        return author_timestamp + text + picture
