"""Local source selection and provenance. This module never calls a model."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from mockingbird.conversation.history import History
from mockingbird.preprocessing.telegram.processor import TelegramProcessor


def pair_id(user: str, other: str) -> str:
    return hashlib.sha256(json.dumps([user, other]).encode()).hexdigest()[:12]


def participant_ids(config: dict) -> dict[str, str]:
    settings = config.get("preprocessing", {})
    user_id, other_id = settings.get("user_id"), settings.get("other_id")
    if user_id is None and other_id is None:
        return {}
    if not user_id or not other_id or user_id == other_id:
        raise ValueError("Set both distinct preprocessing.user_id and other_id")
    return {user_id: config["user"], other_id: config["other"]}


def history_info(history: History, timezone: str = "Europe/Moscow") -> dict:
    zone = ZoneInfo(timezone)
    messages = sorted(
        history.messages,
        key=lambda m: (
            m.timestamp.replace(tzinfo=zone)
            if m.timestamp.tzinfo is None
            else m.timestamp.astimezone(zone)
        ),
    )
    samples = [m.to_sample() for m in messages]
    return {
        "pair": [history.user, history.other],
        "messages": len(messages),
        "first": messages[0].timestamp.isoformat() if messages else None,
        "last": messages[-1].timestamp.isoformat() if messages else None,
        "sha256": hashlib.sha256(
            json.dumps(samples, ensure_ascii=False).encode()
        ).hexdigest(),
    }


def latest_history(
    history: History, count: int, timezone: str = "Europe/Moscow"
) -> History:
    if count < 1:
        raise ValueError("recent message limit must be positive")
    zone = ZoneInfo(timezone)
    messages = sorted(
        (m for m in history.messages if m.author in (history.user, history.other)),
        key=lambda m: (
            m.timestamp.replace(tzinfo=zone)
            if m.timestamp.tzinfo is None
            else m.timestamp.astimezone(zone)
        ),
    )
    return History(history.user, history.other, messages[-count:])


def load_latest_history(config: dict) -> tuple[History, str]:
    """Use the configured raw export when present; otherwise use the cache.

    This does not rewrite the cache during chat startup or call vision APIs.
    The date range is surfaced to the user so an old export cannot look current.
    """
    paths = config["paths"]
    settings = config.get("preprocessing", {})
    source = Path(paths["raw"]) / "result.json"
    if settings.get("latest_messages") is not None and source.exists():
        history = TelegramProcessor(author_ids=participant_ids(config)).process(
            config["user"], config["other"], source.parent
        )
        origin = str(source)
        cached_path = Path(paths["processed"]) / paths["history_filename"]
        if cached_path.exists():
            cached = History.load(cached_path)
            if (cached.user, cached.other) != (config["user"], config["other"]):
                raise ValueError(
                    "Cached history does not match configured participants"
                )
            zone = ZoneInfo(
                config.get("scheduler", {}).get("timezone", "Europe/Moscow")
            )

            def latest(items):
                return max(
                    (
                        (
                            m.timestamp.replace(tzinfo=zone)
                            if m.timestamp.tzinfo is None
                            else m.timestamp.astimezone(zone)
                        )
                        for m in items
                    ),
                    default=datetime.min.replace(tzinfo=zone),
                )

            if latest(cached.messages) > latest(history.messages):
                history = cached
                origin = (
                    str(cached_path) + " (raw export is older or has no pair messages)"
                )
    else:
        source = Path(paths["processed"]) / paths["history_filename"]
        history = History.load(source)
        origin = str(source)
    if (history.user, history.other) != (config["user"], config["other"]):
        raise ValueError("History does not match configured participants")
    limit = settings.get("latest_messages")
    if limit is not None:
        history = latest_history(
            history, limit, config.get("scheduler", {}).get("timezone", "Europe/Moscow")
        )
    return history, origin


def select_persona_history(history: History, config: dict) -> History:
    settings = config.get("persona_build", {})
    selected = latest_history(
        history,
        settings.get("recent_messages", 1000),
        config.get("scheduler", {}).get("timezone", "Europe/Moscow"),
    )
    max_chars = settings.get("max_input_chars", 160000)
    if max_chars < 1:
        raise ValueError("persona_build.max_input_chars must be positive")
    # Preserve complete messages; never cut an author's statement midway.
    total = 0
    recent = []
    for message in reversed(selected.messages):
        size = len(message.to_sample()) + 1
        if total + size > max_chars:
            break
        recent.append(message)
        total += size
    if not recent:
        raise ValueError("No recent source messages fit the persona build budget")
    return History(history.user, history.other, list(reversed(recent)))


def load_manifest(processed: Path) -> dict:
    path = processed / "persona_sources.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def save_manifest(processed: Path, data: dict) -> None:
    processed.mkdir(parents=True, exist_ok=True)
    target = processed / "persona_sources.json"
    temporary = target.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(target)
