from typing import cast

from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

from mockingbird.datacls import ConversationHistory, Response


class Qwen:
    def __init__(
        self,
        model: str = "qwen3-4b-instruct-2507",
        host: str = "http://localhost:1234/v1",
        api_key: str = "unused",
    ):
        self.client = OpenAI(base_url=host, api_key=api_key)

        self.model = model

    def generate(self, history: ConversationHistory, *args, **kwargs) -> Response:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=cast(list[ChatCompletionMessageParam], history),
            **kwargs,
        )

        return Response(response.choices[0].message.content)
