from pathlib import Path

from mockingbird.config import load_config
from mockingbird.llm.client import LLM
from mockingbird.llm.message import Message, Role
from mockingbird.prompts.imitation import IMITATION
from mockingbird.prompts.supress_llm_behavior import SUPRESS_LLM_BEHAVIOR


class Mockingbird:
    def __init__(self):
        config = load_config()

        user = config["user"]

        other = config["other"]

        processed_path = config["paths"]["processed"]
        conversation_folder = f"{user}_{other}".replace(" ", "_")
        inner_prompt_path: Path = (
            processed_path / conversation_folder / "inner_prompt.txt"
        )

        inner_prompt = inner_prompt_path.read_text(encoding="utf-8")
        system_prompt = IMITATION + inner_prompt + SUPRESS_LLM_BEHAVIOR

        llm_config = config["llm"]
        self.model = LLM(llm_config["host"], llm_config["model"], llm_config["api_key"])
        self.inference_temperature = llm_config["inference_temperature"]

        self.conversation_history: list[Message] = []
        self.conversation_history.append(
            Message(role=Role.SYSTEM, content=system_prompt)
        )

    def mock(self):
        while True:
            user_input = input("user>")

            self.conversation_history.append(
                Message(role=Role.USER, content=user_input)
            )

            response = self.model.complete(
                self.conversation_history, temperature=self.inference_temperature
            )

            self.conversation_history.append(
                Message(role=Role.ASSISTANT, content=response)
            )

            print(f"mockingbird>{response}")
