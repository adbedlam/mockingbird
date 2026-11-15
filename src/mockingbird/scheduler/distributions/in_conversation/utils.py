from typing import Iterator

from mockingbird.conversation.history import History
from mockingbird.conversation.message import Message
from mockingbird.conversation.message_type import MessageType


def get_conversation_bursts(history: History) -> Iterator[list[Message]]:
    burst: list[Message] = []

    for message in history.messages:
        is_user_conversation_message = (
            message.author_id == history.user_id
            and message.message_type == MessageType.IN_CONVERSATION_MESSAGE
        )

        if is_user_conversation_message:
            burst.append(message)
        else:
            if burst:
                yield burst
                burst = []

    if burst:
        yield burst