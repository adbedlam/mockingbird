import asyncio
import logging
from pathlib import Path
from datetime import timedelta

import mockingbird.conversation.message as conversation
import mockingbird.llm.message as llm
from mockingbird.config import load_config
from mockingbird.conversation.history import History
from mockingbird.interface.cli import CLIInterface
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

        self.diary = Diary.load(
            processed_path / paths["diary_filename"]
        )

        self.history = History.load(
            processed_path / paths["history_filename"]
        )

    async def sing(self) -> None:
        scheduler_iter = self.scheduler.run().__aiter__()

        scheduler_task = asyncio.create_task(anext(scheduler_iter))
        interface_task = asyncio.create_task(self.interface.receive_event())

        while True:
            done, _ = await asyncio.wait(
                [scheduler_task, interface_task],
                return_when=asyncio.FIRST_COMPLETED,
            )

            if scheduler_task in done:
                event = scheduler_task.result()
                await self._handle_event(event)

                scheduler_task = asyncio.create_task(anext(scheduler_iter))

            if interface_task in done:
                event = interface_task.result()
                await self._handle_event(event)

                interface_task = asyncio.create_task(self.interface.receive_event())

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
            llm.Message(role=Role.SYSTEM, content=IMITATION.format(user=self.user)),
            self.inner_prompt.to_llm_message(),
        ]

        self.context.extend(inner_prompt_context)

        supress_llm_behavior_context = llm.Message(
            role=Role.SYSTEM, content=SUPRESS_LLM_BEHAVIOR
        )

        self.context.append(supress_llm_behavior_context)

        diary_context: list[llm.Message] = [
            llm.Message(role=Role.SYSTEM, content=DIARY_PROMPT),
            self.diary.last_n(self.diary_last_n),
        ]

        self.context.extend(diary_context)

        history_context = self.history.to_llm_messages(self.history_last_n)

        self.context.extend(history_context)

    def _go_to_sleep(self) -> None:
        self.scheduler.generate_schedule()
        print("mockingbird went to sleep")

    def _become_busy(self) -> None:
        print("mockingbird is busy")

    def _become_available(self) -> None:
        print("mockingbird is available")


def main() -> None:
    Mockingbird()


if __name__ == "__main__":
    main()
