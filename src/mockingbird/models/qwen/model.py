from typing import cast

from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

from src.mockingbird.datacls import Message

class Qwen:
    def __init__(
        self, examples: list[Message], model: str = "qwen3-1.7b", base_url: str = "http://localhost:1234/v1"
    ):
        self.history: list[dict[str, str]] = [self._build_system_message(examples)]

        self.client = OpenAI(base_url=base_url, api_key="not-needed")

        self.model = model

    def _build_system_message(self, examples: list[Message]) -> dict[str, str]:
        # This should NOT be hardcoded. However
        instructions = (
            "Your name is Mockingbird."
            "You are to imitate the conversational style of a specific person, "
            "based on the example messages provided below. Your goal is to write "
            "replies that a reader familiar with this person would believe were "
            "written by them.\n\n"
            "Match their length, punctuation, capitalization, vocabulary, emoji use, "
            "and tone. If their messages are short, yours should be too. Do not "
            "use a register the examples don't support. Reply only as this person — "
            "no meta-commentary, no preamble.\n\n"
            "Examples:"
        )
        examples_block = "\n".join(f"- {example.text}" for example in examples)
        return {
                "role": "system",
                "content": f"{instructions}\n{examples_block}"
            }
        
    def _generate(self, messages: list[dict[str, str]], *args, **kwargs):
        return self.client.chat.completions.create(
            model=self.model,
            messages=cast(list[ChatCompletionMessageParam], messages),
            **kwargs
        )

    def converse(self) -> None:
        while True:
            try:
                user_input = input("user> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nbye")
                break

            if not user_input:
                continue

            self.history.append({"role": "user", "content": user_input})

            try:
                response = self._generate(self.history) # why do I pass a member? idk
            except Exception as exception:
                print(f"[error] {exception}")

                self.history.pop()

                continue
            # print(response)
            reply = response.choices[0].message.content or ""
            self.history.append({"role": "assistant", "content": reply})

            print(f"mockingbird> {reply}")


if __name__ == "__main__":
    from src.mockingbird.preprocessing.telegram.processor import TelegramProcessor

    processor = TelegramProcessor()

    from pathlib import Path
    messages = processor.process(Path("data/raw/telegram/Username.json"), "Username")

    import random
    qwen = Qwen(random.sample(messages, k=400), model="qwen3-4b-instruct-2507")

    qwen.converse()

