from enum import StrEnum
from typing import TypedDict


class Role(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class Message(TypedDict):
    role: Role
    content: str
