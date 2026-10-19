import re
from collections import Counter

from mockingbird.conversation import message as conversation
from mockingbird.llm import message as llm
from mockingbird.llm.client import LLM
from mockingbird.llm.message import Role


def tokenize(text: str) -> list[str]:
    return re.findall(r"[а-яёa-z0-9]+", text.lower())


def ngrams(tokens: list[str], n: int) -> list[str]:
    return [" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def top_ngrams(
    messages: list[conversation.Message], n: int = 2, top_k: int = 30
) -> list[str]:
    frequencies: Counter[str] = Counter()

    for message in messages:
        frequencies.update(ngrams(tokenize(message.text), n))

    return [ngram for ngram, _ in frequencies.most_common(top_k)]


NGRAM_INTERESTING_PROMPT = """
You judge whether an n-gram from a person's chat messages is a distinctive marker
of their voice, or generic filler. This is used to profile one speaker, so judge
"interesting" as "would help identify or imitate THIS person", not "semantically
rich".

Interesting: slang, catchphrases, verbal tics, idiosyncratic collocations,
signature openers/closers, personal nicknames, distinctive misspellings.
Boring: function-word pairs, generic grammatical glue, common confirmations,
stock phrases that appear in almost any chat in this language.

Be strict. When unsure, call it boring.

Reply with exactly one word, uppercase: INTERESTING or BORING.
"""


def _is_interesting_ngram(ngram: str, model: LLM) -> bool:
    prompt: list[llm.Message] = [
        {"role": Role.SYSTEM, "content": NGRAM_INTERESTING_PROMPT},
        {"role": Role.USER, "content": ngram},
    ]
    raw = model.complete(prompt, temperature=0.0, max_tokens=4)
    return raw.strip().upper().startswith("INTERESTING")


def remove_boring_ngrams(ngrams: list[str], model: LLM) -> list[str]:
    return [ngram for ngram in ngrams if _is_interesting_ngram(ngram, model)]
