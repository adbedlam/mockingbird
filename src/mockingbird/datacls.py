from dataclasses import dataclass
from datetime import datetime


@dataclass
class Message:
    text: str
    timestamp: datetime
    chat_id: str
    user: str
    user_id: int
