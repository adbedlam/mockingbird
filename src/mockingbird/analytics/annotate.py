from __future__ import annotations

import json
import math
import re
import time
from typing import Any

import pandas as pd

from mockingbird.analytics.prompts.annotate import (
    COMMON_PROMPT,
    CONFRONTATIONAL_PROMPT,
    EMOTION_PROMPT,
    SARCASM_PROMPT,
    TOPIC_PROMPT,
)
from mockingbird.conversation.history import History
from mockingbird.llm import message as llm
from mockingbird.llm.client import LLM
from mockingbird.llm.message import Role

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)

FEATURES = (
    ("topic", TOPIC_PROMPT, ("topic",)),
    ("emotion", EMOTION_PROMPT, ("valence", "arousal")),
    ("sarcasm", SARCASM_PROMPT, ("sarcastic",)),
    ("confrontational", CONFRONTATIONAL_PROMPT, ("confrontational",)),
)


def _system_prompt(
    feature_prompt: str,
    fields: tuple[str, ...],
) -> str:
    schema = {
        "id": "copy the input message ID exactly",
        **{
            field: {
                "topic": "non-empty string",
                "valence": "number from -1 to 1",
                "arousal": "number from -1 to 1",
                "sarcastic": "boolean",
                "confrontational": "boolean",
            }[field]
            for field in fields
        },
    }

    return (
        f"{COMMON_PROMPT}\n"
        f"Required output object schema:\n"
        f"{json.dumps(schema, separators=(',', ':'))}\n\n"
        f"Task-specific instructions:\n{feature_prompt}"
    )


def _format_messages(history: History) -> str:
    messages = [
        {
            "id": message.id,
            "author": str(message.author),
            "timestamp": str(message.timestamp),
            "text": message.text,
        }
        for message in history.messages
    ]

    return json.dumps(
        messages,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _extract_json(text: str) -> Any:
    match = _FENCE_RE.search(text)
    if match:
        text = match.group(1)

    start = text.find("[")
    end = text.rfind("]")

    if start == -1 or end < start:
        raise ValueError(f"no JSON array found in LLM output: {text[:200]!r}")

    return json.loads(text[start : end + 1])


def _validate_value(
    record: dict[str, Any],
    index: int,
    field: str,
) -> None:
    if field not in record:
        raise ValueError(f"record {index} missing field {field!r}")

    value = record[field]

    if field == "topic":
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"record {index} has invalid topic: {value!r}")

    elif field in ("valence", "arousal"):
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
            or not -1.0 <= value <= 1.0
        ):
            raise ValueError(f"record {index} has invalid {field}: {value!r}")

    elif field in ("sarcastic", "confrontational"):
        if not isinstance(value, bool):
            raise ValueError(f"record {index} has invalid {field}: {value!r}")


def _parse(
    text: str,
    expected_ids: list[str],
    fields: tuple[str, ...],
) -> list[dict[str, Any]]:
    records = _extract_json(text)

    if not isinstance(records, list):
        raise ValueError("expected a JSON array")

    expected = set(expected_ids)
    seen: set[str] = set()
    by_id: dict[str, dict[str, Any]] = {}

    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"record {index} is not an object: {record!r}")

        message_id = record.get("id")

        if not isinstance(message_id, str) or message_id not in expected:
            raise ValueError(f"record {index} has unexpected ID: {message_id!r}")

        if message_id in seen:
            raise ValueError(f"duplicate message ID: {message_id!r}")

        seen.add(message_id)

        for field in fields:
            _validate_value(record, index, field)

        by_id[message_id] = record

    missing = expected - seen
    if missing:
        raise ValueError(f"missing {len(missing)} message IDs: {sorted(missing)}")

    unexpected_count = len(records) - len(expected_ids)
    if unexpected_count:
        raise ValueError(f"expected {len(expected_ids)} records, got {len(records)}")

    # The model may reorder records; restore input order.
    return [by_id[message_id] for message_id in expected_ids]


def _batch_ranges(
    n: int,
    batch_size: int,
    overlap: int,
) -> list[tuple[int, int]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not 0 <= overlap < batch_size:
        raise ValueError("overlap must satisfy 0 <= overlap < batch_size")

    step = batch_size - overlap
    return [(start, min(start + batch_size, n)) for start in range(0, n, step)]


def _annotate_batch(
    history: History,
    model: LLM,
    prompt: str,
    fields: tuple[str, ...],
    *,
    retries: int,
) -> list[dict[str, Any]]:
    expected_ids = [message.id for message in history.messages]
    user_content = _format_messages(history)
    system_content = _system_prompt(prompt, fields)
    feedback = ""

    for attempt in range(retries + 1):
        response = model.complete(
            [
                llm.Message(role=Role.SYSTEM, content=system_content),
                llm.Message(
                    role=Role.USER,
                    content=(user_content + feedback),
                ),
            ],
            temperature=0.0,
        )

        try:
            records = _parse(response, expected_ids, fields)
            if attempt:
                print(
                    f"[annotate] Valid response on retry " f"{attempt}/{retries}",
                    flush=True,
                )
            return records

        except (ValueError, json.JSONDecodeError) as exc:
            print(
                f"[annotate] Validation failed on attempt "
                f"{attempt + 1}/{retries + 1}: {exc}",
                flush=True,
            )
            print(
                f"[annotate] Response length: {len(response)}; "
                f"preview: {response[:200]!r}",
                flush=True,
            )

            if attempt == retries:
                raise

            feedback = (
                "\n\nYour previous response failed validation.\n"
                f"Validation error: {exc}\n"
                f"Return exactly {len(expected_ids)} objects, one for "
                "each input ID. Include every ID exactly once. "
                "Do not change IDs. Return only the JSON array."
            )

    raise RuntimeError("unreachable")


def _annotate_feature(
    history: History,
    model: LLM,
    prompt: str,
    fields: tuple[str, ...],
    *,
    batch_size: int,
    overlap: int,
    retries: int,
) -> list[dict[str, Any]]:
    messages = history.messages
    n = len(messages)
    ranges = _batch_ranges(n, batch_size, overlap)
    results: list[dict[str, Any] | None] = [None] * n
    feature_name = ", ".join(fields)

    print(
        f"[annotate] Starting '{feature_name}': "
        f"{n} messages, {len(ranges)} batches",
        flush=True,
    )

    for batch_index, (start, end) in enumerate(ranges):
        started = time.perf_counter()

        print(
            f"[annotate] [{feature_name}] "
            f"Batch {batch_index + 1}/{len(ranges)} "
            f"(messages {start}:{end}, {end - start} messages)...",
            flush=True,
        )

        batch_history = History(
            user=history.user,
            other=history.other,
            messages=messages[start:end],
        )

        try:
            records = _annotate_batch(
                batch_history,
                model,
                prompt,
                fields,
                retries=retries,
            )
        except Exception as exc:
            elapsed = time.perf_counter() - started
            print(
                f"[annotate] [{feature_name}] "
                f"Batch {batch_index + 1}/{len(ranges)} FAILED "
                f"after {elapsed:.1f}s: {type(exc).__name__}: {exc}",
                flush=True,
            )
            raise

        retain_end = (
            ranges[batch_index + 1][0] if batch_index + 1 < len(ranges) else end
        )

        for message_index in range(start, retain_end):
            results[message_index] = records[message_index - start]

        print(
            f"[annotate] [{feature_name}] "
            f"Batch {batch_index + 1}/{len(ranges)} completed "
            f"in {time.perf_counter() - started:.1f}s",
            flush=True,
        )

    if any(record is None for record in results):
        missing = [i for i, record in enumerate(results) if record is None]
        raise RuntimeError(f"messages were not annotated: {missing}")

    print(f"[annotate] Finished '{feature_name}'", flush=True)

    return [record for record in results if record is not None]


def annotate(
    history: History,
    model: LLM,
    *,
    batch_size: int = 40,
    overlap: int = 5,
    retries: int = 2,
) -> pd.DataFrame:
    if retries < 0:
        raise ValueError("retries must be non-negative")

    if not history.messages:
        return pd.DataFrame(
            columns=[
                "topic",
                "valence",
                "arousal",
                "sarcastic",
                "confrontational",
            ]
        )

    if len({message.id for message in history.messages}) != len(history.messages):
        raise ValueError("message IDs must be unique within a history")

    annotations: dict[str, list[dict[str, Any]]] = {}

    for feature_name, prompt, fields in FEATURES:
        annotations[feature_name] = _annotate_feature(
            history,
            model,
            prompt,
            fields,
            batch_size=batch_size,
            overlap=overlap,
            retries=retries,
        )

    topic, emotion, sarcasm, confrontation = (
        annotations["topic"],
        annotations["emotion"],
        annotations["sarcasm"],
        annotations["confrontational"],
    )

    return pd.DataFrame(
        {
            "topic": [record["topic"] for record in topic],
            "valence": [record["valence"] for record in emotion],
            "arousal": [record["arousal"] for record in emotion],
            "sarcastic": [record["sarcastic"] for record in sarcasm],
            "confrontational": [record["confrontational"] for record in confrontation],
        }
    )
