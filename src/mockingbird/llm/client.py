import logging
from typing import cast

from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

from mockingbird.llm.message import Message

logger = logging.getLogger(__name__)


class LLM:
    def __init__(
        self,
        host: str,
        model: str,
        api_key: str | None = None,
        temperature: float = 0.5,
    ) -> None:
        self.model = model

        api_key = api_key

        if api_key is None and not host.startswith(
            ("http://localhost", "http://127.0.0.1")
        ):
            raise ValueError(
                f"Host {host!r} looks remote but no API key was given, set LLM_API_KEY."
            )

        self.client = OpenAI(
            base_url=host,
            api_key=api_key or "not-needed",
        )

        self.temperature = temperature

        logger.debug(
            "Initialized LLM: model=%s, host=%s, temperature=%s",
            model,
            host,
            temperature,
        )

    def complete(self, messages: list[Message]) -> str:
        logger.debug("Sending completion request: model=%s", self.model)
        logger.debug("Request messages (%d): %s", len(messages), messages)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=cast(list[ChatCompletionMessageParam], messages),
                temperature=self.temperature,
            )

            content = response.choices[0].message.content or ""

            logger.debug(
                "Completion received: model=%s, tokens=%s, finish_reason=%s",
                response.model,
                response.usage,
                response.choices[0].finish_reason,
            )
            logger.debug("Response content: %s", content)

            return content

        except Exception:
            logger.exception("LLM completion request failed")
            raise
