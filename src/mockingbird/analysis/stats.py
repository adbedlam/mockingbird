import re
from collections import Counter
from datetime import timedelta

import emoji
import numpy as np

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


def group_by_user(messages: list[Message]) -> dict[int, list[Message]]:
    groups: dict[int, list[Message]] = {}

    for mes in messages:
        groups.setdefault(mes.user_id, []).append(mes)
    return groups


def avg_message_len(messages: list[Message]) -> float:
    if not messages:
        return 0.0
    return sum(len(mes.text) for mes in messages) / len(messages)


def avg_message_len_by_user(messages: list[Message]) -> dict[int, float]:
    return {
        user_id: avg_message_len(mes)
        for user_id, mes in group_by_user(messages=messages).items()
    }


def message_cnt_by_user(messages: list[Message]) -> dict[int, int]:
    counts: dict[int, int] = {}

    for mes in messages:
        counts[mes.user_id] = counts.get(mes.user_id, 0) + 1
    return counts


def emoji_frequency(messages: list[Message]) -> float:
    if not messages:
        return 0.0
    return sum(emoji.emoji_count(mes.text) for mes in messages) / len(messages)


def emoji_frequency_by_user(messages: list[Message]) -> dict[int, float]:
    return {
        user_id: emoji_frequency(mes)
        for user_id, mes in group_by_user(messages).items()
    }


def most_common_emojis(
    messages: list[Message], top_k: int = 5
) -> list[tuple[str, int]]:
    counts: dict[str, int] = {}
    for mes in messages:
        for match in emoji.emoji_list(mes.text):
            entity = match["emoji"]
            counts[entity] = counts.get(entity, 0) + 1

    return sorted(counts.items(), key=lambda x: x[1], reverse=True)[:top_k]


def most_common_emojis_by_user(messages: list[Message]):
    return {
        user_id: most_common_emojis(mes)
        for user_id, mes in group_by_user(messages).items()
    }


def burst_len_by_user(messages: list[Message]) -> dict[int, list[int]]:
    sorted_mes = sorted(messages, key=lambda x: x.timestamp)

    res: dict[int, list[int]] = {}
    current_user = None
    current_len = 0

    for mes in sorted_mes:
        if mes.user_id == current_user:
            current_len += 1
        else:
            if current_user is not None:
                res.setdefault(current_user, []).append(current_len)
            current_user = mes.user_id
            current_len = 1

    if current_user is not None:
        res.setdefault(current_user, []).append(current_len)

    return res


def stat_burst_user(burst: dict[int, list[int]]) -> dict[int, tuple[float, float]]:
    return {
        user_id: (float(np.max(lens)), float(np.mean(lens)))
        for user_id, lens in burst.items()
    }


def response_latency_by_user(messages: list[Message]) -> dict[int, list[timedelta]]:
    sorted_mes = sorted(messages, key=lambda x: x.timestamp)

    res: dict[int, list[timedelta]] = {}

    for prev, curr in zip(sorted_mes, sorted_mes[1:]):
        if curr.user_id != prev.user_id:
            delta = curr.timestamp - prev.timestamp
            res.setdefault(curr.user_id, []).append(delta)

    return res


def stat_response_latency_by_user(
    latency: dict[int, list[timedelta]],
) -> dict[int, tuple[float, float]]:
    return {
        user_id: (
            np.median([d.total_seconds() for d in deltas]),
            np.mean([d.total_seconds() for d in deltas]),
        )
        for user_id, deltas in latency.items()
    }


def tokenize(text: str) -> list[str]:
    return [w for w in re.findall(r"[а-яё]+", text.lower()) if w not in STOPWORDS]


def ngrams(tokens: list[str], n: int) -> list[str]:
    return [" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def ngram_frequency_by_user(
    messages: list[Message], n: int = 2
) -> dict[int, Counter[str]]:
    res: dict[int, Counter[str]] = {}

    for user_id, mes in group_by_user(messages).items():
        counter = Counter[str]()
        for m in mes:
            counter.update(ngrams(tokenize(m.text), n))

        res[user_id] = counter
    return res


def distinctive_ngrams(
    freq_a: Counter[str], freq_b: Counter[str], top_k: int = 15
) -> list[tuple[str, float]]:

    total_a, total_b = sum(freq_a.values()), sum(freq_b.values())
    scores = {}
    for phrase, count_a in freq_a.items():
        rate_a = count_a / total_a
        rate_b = freq_b.get(phrase, 0) / total_b if total_b else 0
        scores[phrase] = rate_a - rate_b

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
