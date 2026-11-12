import asyncio
import logging
from datetime import timedelta
from pathlib import Path

import mockingbird.conversation.message as conversation
import mockingbird.llm.message as llm
from mockingbird.config import load_config
from mockingbird.conversation.history import History
from mockingbird.llm.client import LLM
from mockingbird.llm.message import Role
from mockingbird.mockingbird.event import Event, EventType
from mockingbird.persona.diary.diary import Diary
from mockingbird.persona.inner.inner import InnerPrompt
from mockingbird.preprocessing.factory import make_processor
from mockingbird.preprocessing.vision import VisionModel
from mockingbird.prompts.diary import DIARY_PROMPT
from mockingbird.prompts.imitation import IMITATION
from mockingbird.prompts.supress_llm_behavior import SUPRESS_LLM_BEHAVIOR


class Mockingbird:
    def __init__(self) -> None:
        config = load_config()

        paths = config["paths"]
        processed_path = Path(paths["processed"])

        self.diary = Diary.load(processed_path / paths["diary_filename"])

        self.history = History.load(processed_path / paths["history_filename"])

        self.inner_prompt = InnerPrompt.load(
            processed_path / paths["inner_prompt_filename"]
        )

    async def _handle_event(self, event: Event) -> None:
        match event.type:
            case EventType.RECEIVE_MESSAGE:
                self._receive_message()

            case EventType.SEND_MESSAGE:
                self._send_message()

            case EventType.WAKE_UP:
                self._wake_up()

            case EventType.GO_TO_SLEEP:
                self._go_to_sleep()

            case EventType.BECOME_BUSY:
                self._become_busy()

            case EventType.BECOME_AVAILABLE:
                self._become_available()

    def _send_message(self) -> None:
        pass

    def _receive_message(self) -> None:
        pass

    def _wake_up(self) -> None:
        self._build_context()
        print("mockingbird woke up")

    def _build_context(self) -> None:
        self.context: list[llm.Message] = []

        inner_prompt_context: list[llm.Message] = [
            llm.Message(
                role=Role.SYSTEM, content=IMITATION.format(user=self.history.user)
            ),
            self.inner_prompt.to_llm_message(),
        ]

        self.context.extend(inner_prompt_context)

        supress_llm_behavior_context = llm.Message(
            role=Role.SYSTEM, content=SUPRESS_LLM_BEHAVIOR
        )

        self.context.append(supress_llm_behavior_context)

        # diary_context: list[llm.Message] = [
        #     llm.Message(role=Role.SYSTEM, content=DIARY_PROMPT),
        #     self.diary.last_n(self.diary_last_n),
        # ]

        # self.context.extend(diary_context)

        history_context = self.history.to_llm_messages(self.history.user, 100)

        self.context.extend(history_context)

    def _go_to_sleep(self) -> None:
        print("mockingbird went to sleep")

    def _become_busy(self) -> None:
        print("mockingbird is busy")

    def _become_available(self) -> None:
        print("mockingbird is available")


def main() -> None:
    Mockingbird()


if __name__ == "__main__":
    main()
