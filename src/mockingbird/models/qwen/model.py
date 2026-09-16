from openai import OpenAI

from ..model import Model

class Qwen(Model):
    def __init__(
            self,
            model: str = "qwen3-1.7b",
            base_url: str = "http://localhost:1234/v1"
    ):
        self.client = OpenAI(
            base_url=base_url,
            api_key="not-needed"
        )

        self.model = model

    def generate(
            self,
            messages: list[dict[str, str]],
            *args,
            **kwargs
    ):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            **kwargs
        )

        return response

if __name__ == "__main__":
    qwen = Qwen()

    response = qwen.generate([
        {
            "role": "user",
            "content": "What is 2 + 2?",
        }
    ])

    print(response)