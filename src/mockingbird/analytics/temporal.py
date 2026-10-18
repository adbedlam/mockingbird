from datetime import timedelta
from typing import Any

# this is absolute garbage


def response_latencies(conversation: Any) -> list[timedelta]:
    messages = sorted(conversation.smessages, key=lambda x: x.timestamp)

    latencies: list[timedelta] = []

    for previous, current in zip(messages, messages[1:]):
        if previous.from_other and not current.from_other:
            latencies.append(current.timestamp - previous.timestamp)

    return latencies
