from collections.abc import Iterator
from datetime import timedelta

from mockingbird.conversation.history import History
from mockingbird.conversation.message import Message
from mockingbird.scheduler.observations import conversation_bursts


def get_initiations(
    history: History, conversation_window: timedelta = timedelta(minutes=10)
) -> Iterator[Message]:
    for burst in conversation_bursts(history, conversation_window):
        if burst.author == history.user and burst.initiated:
            yield burst.messages[0]
