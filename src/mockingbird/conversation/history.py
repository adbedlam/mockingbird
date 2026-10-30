from __future__ import annotations

from dataclasses import dataclass

from mockingbird.conversation.message import Message


@dataclass
class History:
    user: str
    other: str
    messages: list[Message]

    def to_samples(self) -> str:
        return "\n".join(message.to_sample() for message in self.messages)
