from dataclasses import dataclass

import httpx2  # type: ignore


@dataclass
class OllamaModel:
    model: str = "llama3.1:8b"
    host: str = "http://localhost:11434"
    timeout: float = 360

    def generate(self, messages: list[dict[str, str]]) -> str:
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
        return response.json()["message"]["content"]
