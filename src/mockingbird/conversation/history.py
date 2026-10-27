from __future__ import annotations

from dataclasses import dataclass

from mockingbird.conversation import message as conversation
from mockingbird.conversation.message import Username


@dataclass
class History:
    user: Username
    other: Username
    messages: list[conversation.Message]

    def to_samples(self) -> str:
        return "\n".join(
            f"{message.author}@{message.timestamp} | {message.text}"
            for message in self.messages
        )
