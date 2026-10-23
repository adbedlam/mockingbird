from pathlib import Path

from mockingbird.conversation.history import History
from mockingbird.llm import message as llm
from mockingbird.llm.client import LLM
from mockingbird.llm.message import Role
from mockingbird.persona.prompts.inner import (
    BUILD_LEXICON_PROMPT,
    EMOJI_FRAMING,
    LEXICON_FRAMING,
    PERSONALITY_PROMPT,
)


def build_personality_prompt(history: History, model: LLM, temperature: float = 0.5):
    messages: list[llm.Message] = [
        llm.Message(role=Role.SYSTEM, content=PERSONALITY_PROMPT),
        llm.Message(role=Role.USER, content=history.to_samples()),
    ]

    personality = model.complete(messages, temperature)

    return personality


def build_lexicon_prompt(ngrams: list[list[str]], model: LLM, temperature: float = 0.5):
    ngram_input = "\n".join(
        f"{ngram}" for ngram in [ngram for group in ngrams for ngram in group]
    )

    messages: list[llm.Message] = [
        llm.Message(role=Role.SYSTEM, content=BUILD_LEXICON_PROMPT),
        llm.Message(role=Role.USER, content=ngram_input),
    ]

    lexicon = model.complete(messages, temperature)

    return LEXICON_FRAMING + lexicon


def build_emoji_prompt(emojis: list[str]) -> str:
    return EMOJI_FRAMING + "\n".join(emojis) if emojis else ""
