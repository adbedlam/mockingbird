from pathlib import Path

import mockingbird.conversation.message as conversation
import mockingbird.llm.message as llm
from mockingbird.config import load_config
from mockingbird.llm.client import LLM
from mockingbird.llm.message import Role
from mockingbird.prompts.imitation import IMITATION
from mockingbird.prompts.supress_llm_behavior import SUPRESS_LLM_BEHAVIOR


class Mockingbird:
    def __init__(self, user: str, other: str, model: LLM, prompt_path: Path):
        self.user = user
        self.other = other

        inner_prompt = prompt_path.read_text(encoding="utf-8")
        system_prompt = IMITATION + inner_prompt + SUPRESS_LLM_BEHAVIOR

        self.model = model

        self.conversation_history: list[llm.Message] = []
        self.conversation_history.append(
            llm.Message(role=Role.SYSTEM, content=system_prompt)
        )

    def feed(self, messages: list[conversation.Message]) -> None:
        conversation_piece = [
            llm.Message(
                role=(Role.ASSISTANT if message.author == self.user else Role.USER),
                content=message.text,
            )
            for message in messages
        ]
        self.conversation_history.extend(conversation_piece)

    def kill(self) -> None:
        self.conversation_history = [
            message
            for message in self.conversation_history
            if message["role"] == Role.SYSTEM
        ]

    def sing(self) -> None:
        while True:
            user_input = input("user>")

            self.conversation_history.append(
                llm.Message(role=Role.USER, content=user_input)
            )

            response = self.model.complete(self.conversation_history)

            self.conversation_history.append(
                llm.Message(role=Role.ASSISTANT, content=response)
            )

            print(f"mockingbird>{response}")

    def chirp(self) -> str:
        return self.model.complete(self.conversation_history)


def main() -> None:
    config = load_config()

    user = config["user"]

    other = config["other"]

    inference_config = config["inference_model"]
    model = LLM(
        inference_config["host"],
        inference_config["model"],
        inference_config["api_key"],
        inference_config["temperature"],
    )

    prompt_path = (
        config["paths"]["processed"]
        / f"{user}_{other}".replace(" ", "_")
        / "inner_prompt.txt"
    )

    mockingbird = Mockingbird(user, other, model, prompt_path)

    mockingbird.sing()


if __name__ == "__main__":
    main()
