import os
from typing import Any, cast

from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

from mockingbird.llm.message import Message


class LLM:
    def __init__(
        self,
        host: str,
        model: str,
        api_key: str | None = None,
    ) -> None:
        self.model = model
        key = api_key or os.getenv("LLM_API_KEY")
        if key is None and not host.startswith(
            ("http://localhost", "http://127.0.0.1")
        ):
            raise ValueError(
                f"Host {host!r} looks remote but no API key was given "
                "(pass api_key= or set LLM_API_KEY)."
            )
        self._client = OpenAI(
            base_url=host,
            api_key=api_key or os.getenv("LLM_API_KEY") or "not-needed",
        )

    def complete(
        self,
        messages: list[Message],
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> str:
        response = self._client.chat.completions.create(
            model=self.model,
            messages=cast(list[ChatCompletionMessageParam], messages),
            temperature=temperature,
            **kwargs,
        )
        return response.choices[0].message.content or ""
