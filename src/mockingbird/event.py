from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any


class EventType(Enum):
    MESSAGE_RECEIVED = auto()
    SEND_MESSAGE = auto()
    WAKE_UP = auto()
    GO_TO_SLEEP = auto()
    BECOME_BUSY = auto()
    BECOME_AVAILABLE = auto()
    END_CONVERSATION = auto()


@dataclass(order=True)
class Event:
    timestamp: datetime
    type: EventType = field(compare=False)
    payload: Any = field(default=None, compare=False)
