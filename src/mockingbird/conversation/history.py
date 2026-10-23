from __future__ import annotations

import re
from dataclasses import dataclass

from mockingbird.conversation import message as conversation
from mockingbird.conversation.username import Username


@dataclass
class History:
    user: Username
    other: Username
    messages: list[conversation.Message]

    def users_messages(self) -> list[conversation.Message]:
        return [message for message in self.messages if message.author == self.user]

    def to_samples(self) -> str:
        return "\n".join(
            f"{message.author}@{message.timestamp} | {message.text}"
            for message in self.messages
        )
