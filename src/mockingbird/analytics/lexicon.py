import re
from collections import Counter

from mockingbird.conversation.history import History


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[а-яёa-z0-9]+", text.lower())


def _ngrams(tokens: list[str], n: int) -> list[str]:
    return [" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def top_ngrams(history: History, n: int = 2, top_k: int = 30) -> list[str]:
    frequencies: Counter[str] = Counter()

    for message in history.messages:
        frequencies.update(_ngrams(_tokenize(message.text), n))

    return [ngram for ngram, _ in frequencies.most_common(top_k)]
