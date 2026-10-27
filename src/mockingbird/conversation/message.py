from dataclasses import dataclass
from datetime import datetime

type Username = str


@dataclass
class Message:
    author: Username
    text: str
    timestamp: datetime
