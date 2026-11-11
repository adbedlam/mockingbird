import asyncio
from datetime import datetime

from mockingbird.conversation.message import Message
from mockingbird.event import Event, EventType

class CLIInterface:
    async def receive_event(self) -> Event:
        text = await asyncio.to_thread(input, "user> ")

        message = Message(
            author="other",
            timestamp=datetime.now(),
            text=text,
            picture_description=None,
            picture_path=None,
            message_id=None,
        )

        return Event(
            timestamp=message.timestamp,
            type=EventType.MESSAGE_RECEIVED,
            payload=message,
        )

    async def send_message(self, message: Message) -> None:
        print(f"mockingbird> {message.text}")