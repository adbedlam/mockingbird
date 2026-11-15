from datetime import timedelta
from collections.abc import Iterator

from mockingbird.conversation.history import History
from mockingbird.conversation.message import Message
from mockingbird.conversation.message_type import MessageType


def get_initiations(history: History, conversation_window: timedelta = timedelta(minutes=10)) -> Iterator[Message]:
    previous = None

    for message in history.messages:
        is_user_initiation = (
            message.author_id == history.user_id
            and message.message_type == MessageType.INITIATION
        )

        if is_user_initiation:
            is_continuation = (
                previous is not None
                and previous.author_id == message.author_id
                and previous.message_type == MessageType.INITIATION
                and timedelta(0)
                <= message.timestamp - previous.timestamp
                <= conversation_window
            )

            if not is_continuation:
                yield message

        previous = message

def get_initiation_bursts(history: History) -> Iterator[list[Message]]:
    burst = []

    for message in history.messages:
        is_user_initiation = (
            message.author_id == history.user_id
            and message.message_type == MessageType.INITIATION
        )

        if not is_user_initiation:
            if burst:
                yield burst
                burst = []
            continue

        is_continuation = (
            burst
            and timedelta(0)
            <= message.timestamp - burst[-1].timestamp
            <= timedelta(minutes=10)
        )

        if is_continuation:
            burst.append(message)
        else:
            if burst:
                yield burst
            burst = [message]

    if burst:
        yield burst