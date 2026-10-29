from pathlib import Path

import mockingbird.conversation.message as conversation
import mockingbird.llm.message as llm
from mockingbird.config import load_config
from mockingbird.llm.client import LLM
from mockingbird.llm.message import Role
from mockingbird.prompts.imitation import IMITATION
from mockingbird.prompts.supress_llm_behavior import SUPRESS_LLM_BEHAVIOR


class Mockingbird:
    def __init__(self):
        config = load_config()

        self.user = config["user"]

        self.other = config["other"]

        processed_path = config["paths"]["processed"]
        conversation_folder = f"{self.user}_{self.other}".replace(" ", "_")
        inner_prompt_path: Path = (
            processed_path / conversation_folder / "inner_prompt.txt"
        )

        inner_prompt = inner_prompt_path.read_text(encoding="utf-8")
        system_prompt = IMITATION + inner_prompt + SUPRESS_LLM_BEHAVIOR

        llm_config = config["llm"]
        self.model = LLM(llm_config["host"], llm_config["model"], llm_config["api_key"])
        self.inference_temperature = llm_config["inference_temperature"]

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

            response = self.model.complete(
                self.conversation_history, temperature=self.inference_temperature
            )

            self.conversation_history.append(
                llm.Message(role=Role.ASSISTANT, content=response)
            )

            print(f"mockingbird>{response}")

    def chirp(self) -> str:
        return self.model.complete(
            self.conversation_history, temperature=self.inference_temperature
        )


def main() -> None:
    mockingbird = Mockingbird()
    mockingbird.sing()


if __name__ == "__main__":
    main()
