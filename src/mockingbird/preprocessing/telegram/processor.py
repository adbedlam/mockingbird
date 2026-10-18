import json
from datetime import datetime
from pathlib import Path

from mockingbird.conversation.history import History
from mockingbird.conversation.message import Message
from mockingbird.conversation.username import Username
from mockingbird.preprocessing.processor import Processor


class TelegramProcessor(Processor):
    def __init__(self):
        pass

    def process(self, user: Username, other: Username, data_path: Path) -> History:
        with data_path.open("r", encoding="utf-8") as file:
            data: dict = json.load(file)

        messages = []
        for message in data["messages"]:
            if (
                not "from" in message
                or "forwarded_from" in message
                or not "text_entities" in message
                or not message["text_entities"]
            ):
                continue

            if not (message["from"] == user or message["from"] == other):
                continue

            timestamp = datetime.fromisoformat(message["date"])

            text = message["text"]

            if isinstance(text, list):
                text = "".join(
                    sent if isinstance(sent, str) else sent.get("text", "")
                    for sent in text
                )

            if not text:
                continue

            messages.append(Message(message["from"], text, timestamp))

        return History(user, other, messages)
