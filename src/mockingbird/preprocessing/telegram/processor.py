import json
from datetime import datetime
from pathlib import Path

from mockingbird.conversation.history import History
from mockingbird.conversation.message import Message, Username
from mockingbird.preprocessing.processor import Processor


class TelegramProcessor(Processor):
    def __init__(self):
        pass

    def process(self, data_path: Path, user: Username, other: Username) -> History:
        with data_path.open("r", encoding="utf-8") as file:
            data: dict = json.load(file)

        messages = []
        for message in data["messages"]:
            if (
                not "from" in message
                or not "text_entities" in message
                or not message["text_entities"]
            ):
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

            if "forwarded_from" in message:
                author = message["forwarded_from"]
            elif message["from"] == user or message["from"] == other:
                author = message["from"]
            messages.append(Message(author, text, timestamp))

        return History(user, other, messages)
