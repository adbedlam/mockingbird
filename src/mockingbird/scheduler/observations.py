"""Observations inferred from timestamps, without mutating imported messages.

A new conversation means a gap longer than conversation_window. This is a
heuristic: without reply/read receipts we cannot know a person's intent.
"""

from collections.abc import Iterator
from dataclasses import dataclass
from datetime import timedelta

from mockingbird.conversation.history import History
from mockingbird.conversation.message import Message


@dataclass
class Burst:
    messages: list[Message]
    initiated: bool

    @property
    def author(self) -> str:
        return self.messages[0].author


def conversation_bursts(
    history: History, conversation_window: timedelta = timedelta(minutes=10)
) -> Iterator[Burst]:
    if conversation_window <= timedelta(0):
        raise ValueError("conversation_window must be positive")
    if history.user == history.other:
        raise ValueError("History participants must be different")
    messages = sorted(
        (m for m in history.messages if m.author in (history.user, history.other)),
        key=lambda m: m.timestamp,
    )
    burst: Burst | None = None
    previous: Message | None = None
    for message in messages:
        silence = (
            previous is not None
            and message.timestamp - previous.timestamp > conversation_window
        )
        if burst is None or message.author != burst.author or silence:
            if burst is not None:
                yield burst
            # The beginning of an export is censored; do not count it as an initiative.
            burst = Burst([message], initiated=silence)
        else:
            burst.messages.append(message)
        previous = message
    if burst is not None:
        yield burst


def response_observations(
    history: History,
    conversation_window: timedelta = timedelta(minutes=10),
    max_delay: timedelta = timedelta(hours=24),
) -> Iterator[tuple[Message, timedelta]]:
    """Incoming burst's last message and delay until the modeled author replies."""
    previous: Burst | None = None
    for burst in conversation_bursts(history, conversation_window):
        if previous is not None and previous.author == history.other:
            if burst.author == history.user:
                incoming = previous.messages[-1]
                delay = burst.messages[0].timestamp - incoming.timestamp
                if timedelta(0) <= delay <= max_delay:
                    yield incoming, delay
        previous = burst
