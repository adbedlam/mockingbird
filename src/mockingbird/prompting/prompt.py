from typing import Protocol

from src.mockingbird.datacls import Message


class Prompt(Protocol):
    def build(
        self, examples: list[Message], user_message: str
    ) -> list[dict[str, str]]: ...
