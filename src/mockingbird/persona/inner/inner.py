from __future__ import annotations

import logging
from dataclasses import dataclass

from mockingbird.conversation.history import History
from mockingbird.llm.client import LLM
from mockingbird.llm.message import Message, Role
from mockingbird.persona.inner.prompts.inner import INNER_PROMPT_COMPONENTS

logger = logging.getLogger(__name__)


@dataclass
class InnerPrompt:
    content: str

    @classmethod
    def from_history(cls, model: LLM, history: History) -> InnerPrompt:
        logger.info(
            "Building inner prompt from %d messages",
            len(history.messages),
        )

        prompt = "\n\n".join(INNER_PROMPT_COMPONENTS).format(
            user=history.user,
            other=history.other,
        )

        messages: list[Message] = [
            Message(role=Role.SYSTEM, content=prompt),
            Message(role=Role.USER, content=history.to_samples()),
        ]

        logger.info("Sending inner prompt generation request to LLM")

        personality = model.complete(messages)

        logger.info(
            "Inner prompt generated: %d characters",
            len(personality),
        )

        if not personality.strip():
            logger.warning("LLM returned an empty inner prompt")

        return cls(content=personality)

    def to_llm_message(self) -> Message:
        return Message(role=Role.SYSTEM, content=self.content)

    # def update(...) -> None:
    # unsure if needed
