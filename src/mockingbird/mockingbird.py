import asyncio
import logging
from datetime import timedelta

from mockingbird.config import load_config
from mockingbird.conversation.history import History
from mockingbird.event import Event, EventType
from mockingbird.llm.client import LLM
import mockingbird.llm.message as llm
from mockingbird.llm.message import Role
import mockingbird.conversation.message as conversation
from mockingbird.persona.diary.diary import Diary
from mockingbird.persona.inner.inner import InnerPrompt
from mockingbird.preprocessing.factory import make_processor
from mockingbird.preprocessing.vision import VisionModel
from mockingbird.scheduler.scheduler import Scheduler
from mockingbird.prompts.imitation import IMITATION
from mockingbird.prompts.supress_llm_behavior import SUPRESS_LLM_BEHAVIOR
from mockingbird.prompts.diary import DIARY_PROMPT
from mockingbird.interface.cli import CLIInterface

class Mockingbird:
    def __init__(self) -> None:
        pass

    def hatch(self) -> None:
        config = load_config()

        self.user = config["user"]
        self.other = config["other"]

        vision_config = config["vision_model"]
        self.vision_model = (
            VisionModel(
                vision_config["host"],
                vision_config["model"],
                vision_config["api_key"],
                vision_config["temperature"],
            )
            if vision_config["enabled"]
            else None
        )

        preprocessor_config = config["preprocessing"]
        processor = make_processor(preprocessor_config["messenger"], self.vision_model)

        data_dir = config["paths"]["raw"]

        self.history: History = processor.process(
            self.user,
            self.other,
            data_dir,
            preprocessor_config["message_start"],
            preprocessor_config["message_end"],
        )

        building_config = config["building_model"]
        self.building_model = LLM(
            building_config["host"],
            building_config["model"],
            building_config["api_key"],
            building_config["temperature"],
        )

        self.inner_prompt = InnerPrompt.from_history(self.building_model, self.history)

        diary_config = config["diary"]
        self.diary = Diary.from_history(
            self.building_model,
            self.history,
            first_n=diary_config["first_n"],
            batch_size=diary_config["batch_size"],
            update_in=config["update_in"],
        )

        scheduler_config = config["scheduler"]
        self.scheduler = Scheduler(
            history=self.history,
            conversation_window=timedelta(
                seconds=scheduler_config["conversation_window"],
            ),
            busy_bin_size=timedelta(
                seconds=scheduler_config["busy"]["bin_size"],
            ),
            busy_max_duration=timedelta(
                seconds=scheduler_config["busy"]["max_duration"],
            ),
            busy_randomness_interval=timedelta(
                seconds=scheduler_config["busy"]["randomness_interval"],
            ),
            burst_window=timedelta(
                seconds=scheduler_config["burst"]["window"],
            ),
            burst_delay_bin_size=timedelta(
                seconds=scheduler_config["burst"]["delay_bin_size"],
            ),
            burst_smoothing_sigma=timedelta(
                seconds=scheduler_config["burst"]["smoothing_sigma"],
            ),
            response_delay_window=timedelta(
                seconds=scheduler_config["response_delay"]["window"],
            ),
            response_delay_bin_size=timedelta(
                seconds=scheduler_config["response_delay"]["bin_size"],
            ),
            response_delay_smoothing_sigma=timedelta(
                seconds=scheduler_config["response_delay"]["smoothing_sigma"],
            ),
            initiation_bin_size=timedelta(
                seconds=scheduler_config["initiation"]["bin_size"],
            ),
            initiation_smoothing_sigma=timedelta(
                seconds=scheduler_config["initiation"]["smoothing_sigma"],
            ),
            sleep_bin_size=timedelta(
                seconds=scheduler_config["sleep"]["bin_size"],
            ),
            sleep_lower_percentile=scheduler_config["sleep"]["lower_percentile"],
            sleep_upper_percentile=scheduler_config["sleep"]["upper_percentile"],
            sleep_start_skew=tuple(scheduler_config["sleep"]["start_skew"]),
            sleep_end_skew=tuple(scheduler_config["sleep"]["end_skew"]),
            sleep_randomness_interval=timedelta(
                seconds=scheduler_config["sleep"]["randomness_interval"],
            ),
        )

        mockingbird_config = config["mockingbird"]
        self.diary_last_n = mockingbird_config["diary_last_n"]
        self.history_last_n = mockingbird_config["history_last_n"]

        self.interface = CLIInterface()

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
            case EventType.MESSAGE_RECEIVED:
                self._send_message()

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

    def _recieve_message(self) -> None:
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

        supress_llm_behavior_context = llm.Message(role=Role.SYSTEM, content=SUPRESS_LLM_BEHAVIOR)

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
    pass

if __name__ == "__main__":
    main()
