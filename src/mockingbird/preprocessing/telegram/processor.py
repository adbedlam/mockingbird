import json
from pathlib import Path

from ..processor import Processor
from ..message import Message

class TelegramProcessor(Processor):
    def __init__(self):
        pass

    def process(self, data_path: Path, username: str) -> list[Message]:
        messages: list[Message] = []

        with data_path.open("r", encoding="utf-8") as f:
            data: dict = json.load(f)

        for message in data["messages"]:
            if (not "from" in message or
                not message["from"] == username or
                not "text_entities" in message or
                not message["text_entities"]
                ): continue

            messages.append(Message(message["text"]))

        return messages
