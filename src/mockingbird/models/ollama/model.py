from dataclasses import dataclass

import httpx2

from mockingbird.models.model import ConversationHistory, Response


@dataclass
class OllamaModel:
    def __init__(
        self,
        model: str = "llama3.1:8b",
        host: str = "http://localhost:11434",
        timeout: float = 360,
    ):
        self.model = model
        self.host = host
        self.timeout = timeout

    def generate(self, messages: ConversationHistory) -> Response:
        response = httpx2.post(
            url=f"{self.host}/api/chat",
            json={
                "model": self.model,
                "messages": messages,
                "stream": False,
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        return Response(response.json()["message"]["content"])
