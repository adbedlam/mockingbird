from typing import Protocol

from mockingbird.llm.message import Message


class CompletionModel(Protocol):
    def complete(self, messages: list[Message]) -> str: ...
