import re
from collections import Counter
from datetime import timedelta

import emoji

from mockingbird.datacls import Message

STOPWORDS = {
    "и",
    "в",
    "не",
    "на",
    "я",
    "с",
    "что",
    "а",
    "то",
    "он",
    "она",
    "как",
    "у",
    "но",
}


def avg_message_len(messages: list[Message]) -> float:
    if not messages:
        return 0.0

    return sum(len(message.text) for message in messages) / len(messages)


def emoji_frequency(messages: list[Message]) -> float:
    if not messages:
        return 0.0

    return sum(emoji.emoji_count(message.text) for message in messages) / len(messages)


def most_common_emojis(
    messages: list[Message], top_k: int = 5
) -> list[tuple[str, int]]:
    counts: Counter[str] = Counter()

    for message in messages:
        for match in emoji.emoji_list(message.text):
            counts[match["emoji"]] += 1

    return counts.most_common(top_k)


def message_bursts(
    messages: list[Message],
    gap: timedelta = timedelta(minutes=3),
) -> list[int]:
    messages = sorted(messages, key=lambda message: message.timestamp)

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


def response_latencies(messages: list[Message]) -> list[timedelta]:
    messages = sorted(messages, key=lambda x: x.timestamp)

    latencies: list[timedelta] = []

    for previous, current in zip(messages, messages[1:]):
        if previous.from_other and not current.from_other:
            latencies.append(current.timestamp - previous.timestamp)

    return latencies


def tokenize(text: str) -> list[str]:
    return [w for w in re.findall(r"[а-яё]+", text.lower()) if w not in STOPWORDS]


def ngrams(tokens: list[str], n: int) -> list[str]:
    return [" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def ngram_frequency(messages: list[Message], n: int = 2) -> Counter[str]:
    frequencies: Counter[str] = Counter()

    for message in messages:
        frequencies.update(ngrams(tokenize(message.text), n))

    return frequencies


def distinctive_ngrams(
    freq_a: Counter[str], freq_b: Counter[str], top_k: int = 15
) -> list[tuple[str, float]]:
    # I don't think this is a very good measure at all
    # comparing against the other user is a very poor baseline
    total_a, total_b = sum(freq_a.values()), sum(freq_b.values())
    scores = {}
    for phrase, count_a in freq_a.items():
        rate_a = count_a / total_a
        rate_b = freq_b.get(phrase, 0) / total_b if total_b else 0
        scores[phrase] = rate_a - rate_b

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
