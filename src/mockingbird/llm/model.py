from enum import StrEnum
from typing import Protocol, TypedDict


class Role(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class Message(TypedDict):
    role: Role
    content: str


class Model(Protocol):
    def complete(self, history: list[Message], *args, **kwargs) -> str: ...
