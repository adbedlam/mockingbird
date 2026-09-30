from dataclasses import dataclass
from datetime import datetime


@dataclass
class Message:
    text: str
    timestamp: datetime
    from_other: bool


@dataclass
class Conversation:
    chat_id: str
    user: str
    messages: list[Message]


type Username = str

type ConversationHistory = list[dict[str, str]]


@dataclass
class Response:
    text: str
