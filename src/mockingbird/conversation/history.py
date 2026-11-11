from __future__ import annotations

from dataclasses import dataclass

import mockingbird.conversation.message as conversation
import mockingbird.llm.message as llm
from mockingbird.llm.message import Role

@dataclass
class History:
    user: str
    other: str
    messages: list[conversation.Message]

    def to_samples(self) -> str:
        return "\n".join(message.to_sample() for message in self.messages)

    def append(self, message: conversation.Message) -> None:
        self.messages.append(message)

    def to_llm_messages(
        self,
        user: str,
        last_n: int,
    ) -> list[llm.Message]:
        messages = self.messages[-last_n:]

        return [
            {
                "role": (
                    Role.ASSISTANT
                    if message.author == user
                    else Role.USER
                ),
                "content": message.to_sample(),
            }
            for message in messages
        ]