import json
from datetime import datetime
from pathlib import Path

from mockingbird.conversation.history import History
from mockingbird.conversation.message import Message
from mockingbird.preprocessing.processor import Processor
from mockingbird.preprocessing.vision import VisionModel


class TelegramProcessor(Processor):
    def __init__(self, vision_model: VisionModel | None = None):
        self.vision_model = vision_model

    def process(
        self,
        user: str,
        other: str,
        data_dir: Path,
        start: int = 0,
        end: int | None = None,
    ) -> History:
        with (data_dir / "result.json").open("r", encoding="utf-8") as file:
            data: dict = json.load(file)

        messages = []
        for message in data["messages"][start:end]:
            author = self._get_author(message)
            timestamp = self._get_timestamp(message)
            text = self._get_text(message)

            relative_picture_path = self._get_picture_path(message)
            picture_path = (
                data_dir / relative_picture_path if relative_picture_path else None
            )
            if picture_path:
                print(picture_path)
            picture_description = (
                self._get_picture_description(picture_path)
                if picture_path and self.vision_model
                else None
            )
            if picture_path:
                print(f"{picture_description}\n")
            if text is None and picture_path is None:
                continue

            messages.append(
                Message(
                    author=author,
                    text=text,
                    picture_description=picture_description,
                    picture_path=picture_path,
                    timestamp=timestamp,
                )
            )

        return History(user, other, messages)

    def _get_author(self, message: dict) -> str:
        if "forwarded_from" in message:
            author = message["forwarded_from"]
        else:
            author = message["from"]

        return author

    def _get_timestamp(self, message: dict) -> datetime:
        return datetime.fromisoformat(message["date"])

    def _get_text(self, message: dict) -> str | None:
        text = message.get("text")

        if not text:
            return None

        if isinstance(text, list):
            return "".join(
                item if isinstance(item, str) else item.get("text", "") for item in text
            )

        return text

    def _get_picture_description(self, path: Path) -> str | None:
        return self.vision_model.describe(path) if self.vision_model else None

    def _get_picture_path(self, message: dict) -> Path | None:
        return message.get("photo")
