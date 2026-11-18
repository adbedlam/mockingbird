import json
import logging
from datetime import datetime
from pathlib import Path

from mockingbird.conversation.history import History
from mockingbird.conversation.message import Message
from mockingbird.preprocessing.processor import Processor
from mockingbird.preprocessing.vision import VisionModel

logger = logging.getLogger(__name__)


class TelegramProcessor(Processor):
    def __init__(
        self,
        vision_model: VisionModel | None = None,
        author_ids: dict[str, str] | None = None,
    ):
        self.vision_model = vision_model
        self.author_ids = author_ids or {}

    def process(
        self,
        user: str,
        other: str,
        data_dir: Path,
        start: int = 0,
        end: int | None = None,
    ) -> History:
        logger.info("Loading Telegram export from %s", data_dir / "result.json")

        with (data_dir / "result.json").open("r", encoding="utf-8") as file:
            data: dict = json.load(file)

        all_messages = data["messages"]
        selected_messages = all_messages[start:end]

        logger.info(
            "Found %d messages in export. Processing messages [%d:%s] (%d selected).",
            len(all_messages),
            start,
            end,
            len(selected_messages),
        )

        messages = []
        pictures_found = 0
        pictures_processed = 0
        skipped = 0

        for index, message in enumerate(selected_messages, start=1):
            author = self._get_author(message)
            if self.author_ids:
                author = self.author_ids.get(message.get("from_id"))

            if (
                author not in (user, other)
                or "forwarded_from" in message
                or message.get("type") == "service"
            ):
                skipped += 1
                logger.debug("Skipping message %d: no author", index)
                continue

            timestamp = self._get_timestamp(message)
            text = self._get_text(message)

            relative_picture_path = self._get_picture_path(message)
            picture_path = (
                data_dir / relative_picture_path if relative_picture_path else None
            )

            picture_description = None

            if picture_path:
                pictures_found += 1

                if self.vision_model:
                    pictures_processed += 1
                    logger.info(
                        "Processing picture %d: %s",
                        pictures_processed,
                        picture_path,
                    )

                    try:
                        picture_description = self._get_picture_description(
                            picture_path
                        )
                    except Exception:
                        logger.exception(
                            "Failed to process picture: %s",
                            picture_path,
                        )

            if text is None and picture_path is None:
                skipped += 1
                logger.debug("Skipping message %d: empty message", index)
                continue

            messages.append(
                Message(
                    author=author,
                    text=text,
                    picture_description=picture_description,
                    picture_path=picture_path,
                    timestamp=timestamp,
                    message_id=message.get("id"),
                )
            )

            logger.debug(
                "Processed message %d/%d: author=%s, timestamp=%s, "
                "text=%s, picture=%s",
                index,
                len(selected_messages),
                author,
                timestamp,
                text is not None,
                picture_path is not None,
            )

            if index % 100 == 0:
                logger.info(
                    "Progress: %d/%d messages processed, %d pictures found, "
                    "%d pictures processed.",
                    index,
                    len(selected_messages),
                    pictures_found,
                    pictures_processed,
                )

        logger.info(
            "Telegram processing complete: "
            "%d messages retained, %d skipped, "
            "%d pictures found, %d pictures processed.",
            len(messages),
            skipped,
            pictures_found,
            pictures_processed,
        )

        return History(user, other, messages)

    def _get_author(self, message: dict) -> str | None:
        return message.get("from")

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
