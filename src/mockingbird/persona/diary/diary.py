from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

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
    entries: list[Entry] = field(default_factory=list)

    @classmethod
    def from_history(
        cls,
        model: LLM,
        history: History,
        first_n=2000,
        batch_size: int = 500,
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
            return cls()

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
            entries=cls._parse_entries(raw_entries),
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
            diary.update(history.user, model, batch)

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

    def update(
        self, user: str, model: LLM, new_messages: list[conversation.Message]
    ) -> None:
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

        raw_entries = model.complete(
            [
                llm.Message(
                    role=Role.SYSTEM,
                    content=DIARY_UPDATE.format(
                        user=user,
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

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)

        data = [
            {
                "timestamp": entry.timestamp.isoformat(),
                "content": entry.content,
            }
            for entry in self.entries
        ]

        path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        logger.info("Diary saved to %s (%d entries)", path, len(self.entries))

    @classmethod
    def load(
        cls,
        path: Path,
    ) -> Diary:
        data = json.loads(path.read_text(encoding="utf-8"))

        entries = [
            Entry(
                timestamp=datetime.fromisoformat(item["timestamp"]),
                content=item["content"],
            )
            for item in data
        ]

        logger.info("Diary loaded from %s (%d entries)", path, len(entries))

        return cls(entries=entries)
