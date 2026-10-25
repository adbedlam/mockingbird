from dataclasses import dataclass, field
from datetime import datetime
from uuid import uuid4

from mockingbird.conversation.username import Username


def _new_message_id() -> str:
    return uuid4().hex[:8]


@dataclass
class Message:
    author: Username
    text: str
    timestamp: datetime
    id: str = field(default_factory=_new_message_id)
