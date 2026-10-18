import re
from collections import Counter

from mockingbird.conversation.message import Message

# should leave removing uninteresting ngrams and words to an LLM


def tokenize(text: str) -> list[str]:
    return re.findall(r"[а-яёa-z0-9]+", text.lower())


def ngrams(tokens: list[str], n: int) -> list[str]:
    return [" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


# History


def top_ngrams(
    messages: list[Message], n: int = 2, top_k: int = 15
) -> list[tuple[str, int]]:
    frequencies: Counter[str] = Counter()

    for message in messages:
        frequencies.update(ngrams(tokenize(message.text), n))

    return frequencies.most_common(top_k)
