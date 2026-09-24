import json
from datetime import datetime
from pathlib import Path

from mockingbird.datacls import Conversation, Message


class TelegramProcessor:
    def __init__(self):
        pass

    def process(self, data_path: Path) -> Conversation:
        with data_path.open("r", encoding="utf-8") as file:
            data: dict = json.load(file)

        conversation = Conversation(chat_id=data["id"], user=data["name"], messages=[])

        for message in data["messages"]:
            if (
                not "from" in message
                or "forwarded_from" in message
                or not "text_entities" in message
                or not message["text_entities"]
            ):
                continue

            timestamp = datetime.fromisoformat(message["date"])
            from_other = message["from"] == data["name"]
            if from_other:
                conversation.messages.append(Message("", timestamp, from_other))
                continue

            text = message["text"]
            if isinstance(text, list):
                text = "".join(
                    sent if isinstance(sent, str) else sent.get("text", "")
                    for sent in text
                )

            if not text:
                continue

            conversation.messages.append(Message(text, timestamp, from_other))

        return conversation
