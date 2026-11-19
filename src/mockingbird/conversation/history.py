from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import mockingbird.conversation.message as conversation
import mockingbird.llm.message as llm
from mockingbird.llm.message import Role


@dataclass
class History:
    user: str
    other: str
    messages: list[conversation.Message]

    def to_samples(self) -> str:
        return "\n".join(message.to_sample() for message in self.messages)

    def append(self, message: conversation.Message) -> None:
        self.messages.append(message)

    def to_llm_messages(
        self,
        user: str,
        last_n: int,
    ) -> list[llm.Message]:
        if last_n < 0:
            raise ValueError("last_n cannot be negative")
        if last_n == 0:
            return []
        messages = self.messages[-last_n:]

        return [
            {
                "role": (Role.ASSISTANT if message.author == user else Role.USER),
                "content": message.to_sample(),
            }
            for message in messages
        ]

    def to_dict(self) -> dict:
        return {
            "user": self.user,
            "other": self.other,
            "messages": [
                {
                    "author": message.author,
                    "timestamp": message.timestamp.isoformat(),
                    "text": message.text,
                    "picture_description": message.picture_description,
                    "picture_path": (
                        str(message.picture_path)
                        if message.picture_path is not None
                        else None
                    ),
                    "message_id": message.message_id,
                }
                for message in self.messages
            ],
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary.replace(path)

    @classmethod
    def load(cls, path: Path) -> History:
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))

    @classmethod
    def from_dict(cls, data: dict) -> History:

        messages = [
            conversation.Message(
                author=message["author"],
                timestamp=datetime.fromisoformat(message["timestamp"]),
                text=message["text"],
                picture_description=message["picture_description"],
                picture_path=(
                    Path(message["picture_path"])
                    if message["picture_path"] is not None
                    else None
                ),
                message_id=message["message_id"],
            )
            for message in data["messages"]
        ]

        return cls(
            user=data["user"],
            other=data["other"],
            messages=messages,
        )

    @classmethod
    def history_info(cls, history: History, timezone: str = "Europe/Moscow") -> dict:
        zone = ZoneInfo(timezone)
        messages = sorted(
            history.messages,
            key=lambda message: (
                message.timestamp.replace(tzinfo=zone)
                if message.timestamp.tzinfo is None
                else message.timestamp.astimezone(zone)
            ),
        )
        samples = [message.to_sample() for message in messages]
        return {
            "pair": [history.user, history.other],
            "messages": len(messages),
            "first": messages[0].timestamp.isoformat() if messages else None,
            "last": messages[-1].timestamp.isoformat() if messages else None,
            "sha256": hashlib.sha256(
                json.dumps(samples, ensure_ascii=False).encode()
            ).hexdigest(),
        }

    @classmethod
    def latest_history(
        cls, history: History, count: int, timezone: str = "Europe/Moscow"
    ) -> History:
        if count < 1:
            raise ValueError("recent message limit must be positive")
        zone = ZoneInfo(timezone)
        messages = sorted(
            (
                message
                for message in history.messages
                if message.author in (history.user, history.other)
            ),
            key=lambda message: (
                message.timestamp.replace(tzinfo=zone)
                if message.timestamp.tzinfo is None
                else message.timestamp.astimezone(zone)
            ),
        )
        return cls(history.user, history.other, messages[-count:])
