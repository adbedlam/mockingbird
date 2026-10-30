from typing import cast

from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

from mockingbird.llm.message import Message


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

    def complete(
        self,
        messages: list[Message],
    ) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=cast(list[ChatCompletionMessageParam], messages),
            temperature=self.temperature,
        )
        return response.choices[0].message.content or ""
