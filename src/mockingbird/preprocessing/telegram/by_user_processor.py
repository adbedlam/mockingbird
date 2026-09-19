import json
from datetime import datetime
from pathlib import Path

from mockingbird.datacls import Message


class UserTelegramProcessor:
    def process(self, data_path: Path) -> list[Message]:
        with data_path.open("r", encoding="utf-8") as file:
            data = json.load(file)

        messages: list[Message] = []
        chat_id = str(data["id"])

        for mes in data["messages"]:
            if (
                mes.get("type") != "message"
                or not mes.get("text")
                or "forwarded_from" in mes
            ):
                continue

            text = mes["text"]

            if isinstance(text, list):
                text = "".join(
                    sent if isinstance(sent, str) else sent.get("text", "")
                    for sent in text
                )

            if not text:
                continue

            user = mes["from_id"]
            user_id = int(user.removeprefix("user"))

            messages.append(
                Message(
                    text=text,
                    timestamp=datetime.fromisoformat(mes["date"]),
                    chat_id=chat_id,
                    user=user,
                    user_id=user_id,
                )
            )

        return messages
