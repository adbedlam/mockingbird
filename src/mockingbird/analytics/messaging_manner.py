from datetime import timedelta
from typing import Any

# sort of ew


def message_bursts(
    conversation: Any,
    gap: timedelta = timedelta(minutes=3),
) -> list[int]:
    messages = sorted(conversation.messages, key=lambda message: message.timestamp)

    if not messages:
        return []

    bursts: list[int] = []
    current_len = 1

    for previous, current in zip(messages, messages[1:]):
        if current.timestamp - previous.timestamp <= gap:
            current_len += 1
        else:
            bursts.append(current_len)
            current_len = 1

    bursts.append(current_len)

    return bursts
