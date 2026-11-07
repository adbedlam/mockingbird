from dataclasses import dataclass
from enum import Enum, auto


class EventType(Enum):
    MESSAGE_RECEIVED = auto()
    SEND_MESSAGE = auto()
    WAKE_UP = auto()
    GO_TO_SLEEP = auto()
    BECOME_BUSY = auto()
    BECOME_AVAILABLE = auto()


@dataclass
class Event:
    type: EventType
    payload: object | None = None
