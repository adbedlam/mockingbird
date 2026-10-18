from collections import Counter

import emoji

from mockingbird.conversation.message import Message


def most_common_emojis(
    messages: list[Message], top_k: int = 5
) -> list[tuple[str, int]]:
    counts: Counter[str] = Counter()

    for message in messages:
        for match in emoji.emoji_list(message.text):
            counts[match["emoji"]] += 1

    return counts.most_common(top_k)


def emoji_frequency(messages: list[Message]) -> float:
    return sum(emoji.emoji_count(message.text) for message in messages) / len(messages)
