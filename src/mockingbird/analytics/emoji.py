from collections import Counter

import emoji

from mockingbird.conversation.history import History
from mockingbird.conversation.message import Message


def most_common_emojis(history: History, top_k: int = 8) -> list[str]:
    counts: Counter[str] = Counter()

    for message in history.users_messages():
        for match in emoji.emoji_list(message.text):
            counts[match["emoji"]] += 1

    return [emoji for emoji, _ in counts.most_common(top_k)]


def emoji_count(message: Message) -> int:
    return emoji.emoji_count(message.text)
