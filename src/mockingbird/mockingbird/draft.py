from __future__ import annotations

import json
import re
from dataclasses import dataclass

from mockingbird.scheduler.scheduler import MessageKind


@dataclass(frozen=True)
class Draft:
    send: bool
    text: str
    reason: str
    format: str = "json"
    life_update: dict | None = None


def parse_draft(raw: str, kind: MessageKind | None) -> Draft:
    raw = raw.strip().lstrip("\ufeff").strip()
    if not raw:
        raise ValueError("Model returned an empty response")
    candidate = raw
    fence = re.fullmatch(
        r"```(?:json)?\s*(.*?)\s*```", raw, flags=re.DOTALL | re.IGNORECASE
    )
    if fence and (
        raw.lower().startswith("```json")
        or fence[1].lstrip().startswith(("{", "["))
    ):
        candidate = fence[1].strip()
    try:
        data = json.loads(candidate)
    except json.JSONDecodeError:
        structured = (
            candidate.startswith(("{", "[", '"'))
            or raw.lower().startswith("```json")
            or re.search(r'"(?:send|text|reason)"\s*:', raw)
        )
        if kind == MessageKind.REPLY and not structured:
            return Draft(True, raw, "", format="plain_reply")
        raise
    if not isinstance(data, dict) or type(data.get("send")) is not bool:
        raise ValueError("Expected a structured send decision")
    text, reason = data.get("text"), data.get("reason")
    if (
        not isinstance(text, str)
        or not isinstance(reason, str)
        or not reason.strip()
    ):
        raise ValueError("Expected text and a visible decision note")
    if data["send"] and not text.strip():
        raise ValueError("Cannot send an empty message")
    if not data["send"] and kind != MessageKind.INITIATIVE:
        raise ValueError("A reply cannot be silently declined")
    update = data.get("life_update")
    if update is not None and (
        not isinstance(update, dict)
        or not data["send"]
        or kind != MessageKind.REPLY
    ):
        raise ValueError("Only a sent reply can include a structured life update")
    return Draft(data["send"], text.strip(), reason.strip(), life_update=update)
