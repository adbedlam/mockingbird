from __future__ import annotations

import json
import re

import pandas as pd

from mockingbird.analytics.prompts.annotate import (
    ANNOTATION_PROMPT,
    RETRY_SUFFIX,
    USER_TEMPLATE,
)
from mockingbird.conversation.history import History
from mockingbird.llm import message as llm
from mockingbird.llm.client import LLM
from mockingbird.llm.message import Role

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def _parse(text: str, n: int) -> list:
    match = _FENCE_RE.search(text)
    if match:
        text = match.group(1)
    start, end = text.find("["), text.rfind("]")
    if start == -1 or end < start:
        raise ValueError(f"no JSON array found in LLM output: {text[:200]!r}")

    records = json.loads(text[start : end + 1])

    if not isinstance(records, list) or len(records) != n:
        raise ValueError(
            f"expected a list of {n}, got {type(records).__name__} of {len(records) if isinstance(records, list) else '?'}"
        )

    for i, record in enumerate(records):
        emotion = record.get("emotion") if isinstance(record, dict) else None
        if not (
            isinstance(record, dict)
            and isinstance(record.get("topic"), str)
            and isinstance(emotion, dict)
            and isinstance(emotion.get("valence"), (int, float))
            and isinstance(emotion.get("arousal"), (int, float))
            and isinstance(record.get("sarcastic"), bool)
            and isinstance(record.get("confrontational"), bool)
        ):
            raise ValueError(f"record {i} malformed: {record!r}")
    return records


def annotate(history: History, model: LLM, *, retries: int = 2) -> pd.DataFrame:
    messages = history.messages
    n = len(messages)
    if n == 0:
        return pd.DataFrame(
            columns=["topic", "valence", "arousal", "sarcastic", "confrontational"]
        )

    user = USER_TEMPLATE.format(n=n, samples=history.to_samples())

    for attempt in range(retries + 1):
        response = model.complete(
            [
                llm.Message(role=Role.SYSTEM, content=ANNOTATION_PROMPT),
                llm.Message(
                    role=Role.USER,
                    content=user if attempt == 0 else user + RETRY_SUFFIX.format(n=n),
                ),
            ],
            temperature=0.0 if attempt == 0 else 0.3,
        )
        try:
            records = _parse(response, n)
        except (ValueError, json.JSONDecodeError):
            if attempt == retries:
                raise
            continue

        return pd.DataFrame(
            {
                "topic": [r["topic"] for r in records],
                "valence": [r["emotion"]["valence"] for r in records],
                "arousal": [r["emotion"]["arousal"] for r in records],
                "sarcastic": [r["sarcastic"] for r in records],
                "confrontational": [r["confrontational"] for r in records],
            }
        )

    raise RuntimeError(
        f"annotate: unreachable — {retries + 1} attempts exhausted without return/raise"
    )
