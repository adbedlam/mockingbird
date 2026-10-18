from dataclasses import dataclass
from datetime import datetime

from mockingbird.conversation.username import Username


@dataclass
class Message:
    author: Username
    text: str
    timestamp: datetime
