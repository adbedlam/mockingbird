import logging
from typing import cast

from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

from mockingbird.llm.message import Message
from mockingbird.llm.protocol import CompletionModel

logger = logging.getLogger(__name__)


class LLM(CompletionModel):
    def __init__(
        self,
        host: str,
        model: str,
        api_key: str | None = None,
        temperature: float = 0.5,
        *,
        timeout: float = 120.0,
        json_mode: bool = False,
    ) -> None:
        self.model = model
        self.json_mode = json_mode

        if not api_key and not host.startswith(
            ("http://localhost", "http://127.0.0.1")
        ):
            raise ValueError(
                f"Host {host!r} requires an API key. Set the corresponding "
                "BUILDING_MODEL_API_KEY, INFERENCE_MODEL_API_KEY or VISION_MODEL_API_KEY."
            )

        self.client = OpenAI(
            base_url=host,
            api_key=api_key or "not-needed",
            timeout=timeout,
            max_retries=0,
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
        logger.debug("Request message count: %d", len(messages))

        try:
            if self.json_mode:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=cast(list[ChatCompletionMessageParam], messages),
                    temperature=self.temperature,
                    response_format={"type": "json_object"},
                )
            else:
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
            logger.debug("Response length: %d", len(content))

            return content

        except Exception as exc:
            logger.warning("LLM completion request failed: %s", type(exc).__name__)
            raise
