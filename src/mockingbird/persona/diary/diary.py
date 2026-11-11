from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime

from mockingbird.conversation import message as conversation
from mockingbird.conversation.history import History
from mockingbird.llm import message as llm
from mockingbird.llm.client import LLM
from mockingbird.llm.message import Role
from mockingbird.persona.diary.entry import Entry
from mockingbird.persona.diary.prompts.diary_generate import DIARY_GENERATE
from mockingbird.persona.diary.prompts.diary_update import DIARY_UPDATE

logger = logging.getLogger(__name__)


@dataclass
class Diary:
    user: str
    model: LLM
    update_in: int
    entries: list[Entry] = field(default_factory=list)
    pending_messages: list[conversation.Message] = field(default_factory=list)
    diary_updated: bool = False

    @classmethod
    def from_history(
        cls,
        model: LLM,
        history: History,
        first_n=2000,
        batch_size: int = 500,
        update_in: int = 200,
    ) -> Diary:
        if first_n < 1:
            raise ValueError("first_n must be positive")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")

        messages = sorted(
            history.messages,
            key=lambda message: message.timestamp,
        )

        total = len(messages)

        if not messages:
            logger.warning("Cannot generate diary: conversation history is empty")
            return cls(user=history.user, model=model, update_in=update_in)

        first_batch = messages[:first_n]

        logger.info(
            "Generating initial diary from %d/%d messages",
            len(first_batch),
            total,
        )

        raw_entries = model.complete(
            [
                llm.Message(
                    role=Role.SYSTEM,
                    content=DIARY_GENERATE.format(
                        user=history.user, other=history.other
                    ),
                ),
                llm.Message(
                    role=Role.USER,
                    content="\n".join(message.to_sample() for message in first_batch),
                ),
            ]
        )

        diary = cls(
            user=history.user,
            entries=cls._parse_entries(raw_entries),
            model=model,
            pending_messages=[],
            update_in=update_in,
        )

        logger.info(
            "Initial diary generated: %d entries",
            len(diary.entries),
        )

        for start in range(first_n, total, batch_size):
            batch = messages[start : start + batch_size]

            logger.info(
                "Updating diary with messages %d-%d/%d",
                start + 1,
                start + len(batch),
                total,
            )

            previous_count = len(diary.entries)
            diary._update(batch)

            logger.info(
                "Diary updated: +%d entries (%d total)",
                len(diary.entries) - previous_count,
                len(diary.entries),
            )

        logger.info(
            "Diary generation complete: %d entries from %d messages",
            len(diary.entries),
            total,
        )

        return diary

    @staticmethod
    def _parse_entries(output: str) -> list[Entry]:
        output = output.strip()

        output = re.sub(r"^```(?:json)?\s*", "", output)
        output = re.sub(r"\s*```$", "", output)

        data = json.loads(output)

        if not isinstance(data, dict):
            raise ValueError("Expected a JSON object.")

        raw_entries = data.get("entries")

        if not isinstance(raw_entries, list):
            raise ValueError("Expected 'entries' to be a list.")

        entries = []

        for index, raw_entry in enumerate(raw_entries):
            if not isinstance(raw_entry, dict):
                raise ValueError(f"Entry {index} must be an object.")

            timestamp = raw_entry.get("timestamp")
            content = raw_entry.get("content")

            if not isinstance(timestamp, str):
                raise ValueError(f"Entry {index}: invalid timestamp.")

            if not isinstance(content, str) or not content.strip():
                raise ValueError(f"Entry {index}: invalid content.")

            try:
                parsed_timestamp = datetime.fromisoformat(timestamp)
            except ValueError as exc:
                raise ValueError(f"Entry {index}: invalid ISO 8601 timestamp.") from exc

            entries.append(
                Entry(
                    timestamp=parsed_timestamp,
                    content=content.strip(),
                )
            )

        return entries

    def __str__(self) -> str:
        return "\n\n".join(
            f"{entry.timestamp.isoformat()} | {entry.content}" for entry in self.entries
        )

    def _update(self, new_messages: list[conversation.Message]) -> None:
        if not new_messages:
            logger.debug("Skipping diary update: no new messages")
            return

        logger.info(
            "Updating diary with %d new messages (%d existing entries)",
            len(new_messages),
            len(self.entries),
        )

        new_conversation = "\n".join(message.to_sample() for message in new_messages)

        existing_diary = str(self)

        raw_entries = self.model.complete(
            [
                llm.Message(
                    role=Role.SYSTEM,
                    content=DIARY_UPDATE.format(
                        user=self.user,
                    ),
                ),
                llm.Message(
                    role=Role.USER,
                    content=(
                        f"EXISTING DIARY:\n{existing_diary}\n\n"
                        f"NEW CONVERSATION:\n{new_conversation}"
                    ),
                ),
            ]
        )

        new_entries = self._parse_entries(raw_entries)
        self.entries.extend(new_entries)
        self.entries.sort(key=lambda entry: entry.timestamp)

        logger.info(
            "Diary update complete: added %d entries (%d total)",
            len(new_entries),
            len(self.entries),
        )

        self.diary_updated = True

    def append_message(self, message: conversation.Message) -> None:
        self.pending_messages.append(message)

        if (len(self.pending_messages)) >= self.update_in:
            self._update(self.pending_messages)
            self.pending_messages.clear()

    def last_n(self, last_n: int) -> list[Entry]:
        return self.entries[-last_n:]

    def check_diary(self) -> bool:
        return self.diary_updated

    def diary_read(self) -> None:
        self.diary_updated = False

def main():
    logging.basicConfig(level=logging.INFO)
    from mockingbird.config import load_config

    config = load_config()

    user = config["user"]
    other = config["other"]
    vision_config = config["vision_model"]
    from mockingbird.preprocessing.vision import VisionModel

    vision_model = (
        VisionModel(
            vision_config["host"],
            vision_config["model"],
            vision_config["api_key"],
            vision_config["temperature"],
        )
        if vision_config["enabled"]
        else None
    )
    from mockingbird.preprocessing.factory import make_processor

    processor = make_processor(
        config["preprocessing"]["messenger"],
        vision_model=vision_model,
    )

    history: History = processor.process(
        user,
        other,
        config["paths"]["raw"],
        0,
        1100,
    )

    building_config = config["building_model"]
    model = LLM(
        building_config["host"], building_config["model"], building_config["api_key"]
    )
    diary = Diary.from_history(
        model,
        history=History(
            user=history.user,
            other=history.other,
            messages=history.messages[:1000],
        ),
        first_n=500,
        batch_size=100,
        update_in=100,
    )
    with open("diary_1.txt", "w", encoding="utf-8") as f:
        f.write(str(diary))

    diary._update(history.messages[1000:1100])

    with open("history.txt", "w", encoding="utf-8") as f:
        f.write(history.to_samples())

    with open("diary_2.txt", "w", encoding="utf-8") as f:
        f.write(str(diary))


if __name__ == "__main__":
    main()
