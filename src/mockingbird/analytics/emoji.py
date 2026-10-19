from collections import Counter

import emoji

from mockingbird.conversation.message import Message


def most_common_emojis(
    messages: list[Message], top_k: int = 8
) -> list[tuple[str, int]]:
    counts: Counter[str] = Counter()

    for message in messages:
        for match in emoji.emoji_list(message.text):
            counts[match["emoji"]] += 1

    return counts.most_common(top_k)


def emoji_frequency(messages: list[Message]) -> float:
    if not messages:
        return 0.0
    return sum(emoji.emoji_count(message.text) for message in messages) / len(messages)


def message_emoji_count(message: Message) -> int:
    return emoji.emoji_count(message.text)
