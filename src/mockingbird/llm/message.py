from enum import StrEnum
from typing import Any, TypedDict


class Role(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class Message(TypedDict):
    role: Role
    content: Any
