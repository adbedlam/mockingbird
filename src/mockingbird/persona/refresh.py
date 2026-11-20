"""Rebuild profile and initial diary from the same latest, labeled evidence."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Protocol
from zoneinfo import ZoneInfo

from mockingbird.conversation.history import History
from mockingbird.llm.message import Message, Role
from mockingbird.persona.diary.diary import Diary
from mockingbird.persona.diary.entry import Entry
from mockingbird.persona.diary.runtime import DiaryMemory
from mockingbird.persona.inner.inner import InnerPrompt
from mockingbird.persona.life import LifePrompt
from mockingbird.persona.prompts.refresh import (
    diary_refresh_prompt,
    life_refresh_prompt,
    local_life_prompt,
    local_profile_prompt,
    profile_refresh_prompt,
)
from mockingbird.persona.source import (
    load_manifest,
    save_manifest,
    select_persona_history,
)
from mockingbird.preprocessing.telegram.source import load_latest_history


class Model(Protocol):
    def complete(self, messages: list[Message]) -> str: ...


@dataclass
class RefreshPlan:
    history: History
    source: dict
    profile_messages: list[Message]
    life_messages: list[Message]
    diary_messages: list[Message]
    model: dict


def archive_source_id(message) -> str:
    return (
        f"message:{message.message_id}"
        if message.message_id is not None
        else DiaryMemory.fingerprint(message)
    )


def plan_refresh(config: dict) -> RefreshPlan:
    history, origin = load_latest_history(config)
    selected = select_persona_history(history, config)
    timezone = config.get("scheduler", {}).get("timezone", "Europe/Moscow")
    source = {
        **History.history_info(selected, timezone),
        "origin": origin,
        "timezone": timezone,
    }
    profile_prompt = profile_refresh_prompt(selected.user, selected.other, source)
    profile_messages = [
        Message(role=Role.SYSTEM, content=profile_prompt),
        Message(role=Role.USER, content=selected.to_samples()),
    ]
    life_prompt = life_refresh_prompt(selected.user, selected.other, source)
    life_messages = [
        Message(role=Role.SYSTEM, content=life_prompt),
        Message(role=Role.USER, content=selected.to_samples()),
    ]
    evidence = [
        {
            "id": archive_source_id(m),
            "author": m.author,
            "timestamp": m.timestamp.isoformat(),
            "text": m.text,
            "picture_description": m.picture_description,
        }
        for m in selected.messages
    ]
    diary_prompt = diary_refresh_prompt(selected.user, selected.other)
    diary_messages = [
        Message(role=Role.SYSTEM, content=diary_prompt),
        Message(role=Role.USER, content=json.dumps(evidence, ensure_ascii=False)),
    ]
    settings = config["building_model"]
    if len({row["id"] for row in evidence}) != len(evidence):
        raise ValueError("Source history contains duplicate message IDs")
    return RefreshPlan(
        selected,
        source,
        profile_messages,
        life_messages,
        diary_messages,
        {
            key: settings[key]
            for key in ("host", "model", "temperature")
            if key in settings
        },
    )


def refresh_needed(config: dict) -> bool:
    processed = Path(config["paths"]["processed"])
    if not all(
        (
            processed
            / (
                config["paths"].get(key, default)
                if key != "inner_prompt_filename"
                else config["paths"][key]
            )
        ).exists()
        for key, default in (
            ("inner_prompt_filename", "inner_prompt.md"),
            ("life_prompt_filename", "life_prompt.md"),
            ("diary_filename", "diary.json"),
        )
    ):
        return True
    history, _ = load_latest_history(config)
    selected = select_persona_history(history, config)
    source = History.history_info(
        selected, config.get("scheduler", {}).get("timezone", "Europe/Moscow")
    )
    manifest = load_manifest(processed)
    return any(
        manifest.get(key, {}).get("sha256") != source["sha256"]
        for key in ("profile", "life", "diary")
    )


def bootstrap_local_persona(config: dict) -> dict:
    """Usable evidence-only starting point when remote profile building is unavailable.

    Does not infer personality or rewrite memories, and never overwrites an existing
    profile. A later explicit model refresh backs these artifacts up as usual.
    """
    processed = Path(config["paths"]["processed"])
    profile_path = processed / config["paths"]["inner_prompt_filename"]
    diary_path = processed / config["paths"]["diary_filename"]
    life_prompt_path = processed / config["paths"].get(
        "life_prompt_filename", "life_prompt.md"
    )
    if profile_path.exists() or diary_path.exists() or life_prompt_path.exists():
        raise FileExistsError("Local bootstrap only creates a new persona")
    history, origin = load_latest_history(config)
    selected = select_persona_history(history, config)
    own = [
        m
        for m in selected.messages
        if m.author == selected.user and m.text and m.text.strip()
    ]
    if not own:
        raise ValueError("No source examples from the modeled person")
    texts = [m.text.strip() for m in own if m.text is not None]
    examples = [m for m in own if len(m.text or "") <= 500][-60:]
    if not examples:
        examples = own[-5:]
    full_stops = round(100 * sum(t.endswith(".") for t in texts) / len(texts), 1)
    commas = round(100 * sum("," in t for t in texts) / len(texts), 1)
    content = local_profile_prompt(
        selected.user,
        selected.other,
        len(texts),
        full_stops,
        commas,
        "\n".join(m.to_sample() for m in examples),
    )
    timezone = config.get("scheduler", {}).get("timezone", "Europe/Moscow")
    source = {
        **History.history_info(selected, timezone),
        "origin": origin,
        "built_at": datetime.now(ZoneInfo(timezone)).isoformat(),
        "model": None,
    }
    InnerPrompt(content).save(profile_path)
    LifePrompt(
        content=local_life_prompt(
            selected.user,
            "\n".join(m.to_sample() for m in examples),
        )
    ).save(life_prompt_path)
    Diary().save(diary_path)
    metadata = {
        "version": 1,
        "profile": {
            **source,
            "method": "local_examples",
            "example_count": len(examples),
        },
        "life": {**source, "method": "local_examples", "example_count": len(examples)},
        "diary": {**source, "method": "local_empty", "entries": 0},
    }
    save_manifest(processed, metadata)
    return metadata


def prepare_refresh(plan: RefreshPlan, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for name, messages in (
        ("profile", plan.profile_messages),
        ("life", plan.life_messages),
        ("diary", plan.diary_messages),
    ):
        payload = {
            "source": plan.source,
            "method": "POST",
            "endpoint": plan.model["host"].rstrip("/") + "/chat/completions",
            "body": {
                "model": plan.model["model"],
                "temperature": plan.model.get("temperature", 0.3),
                "messages": messages,
            },
        }
        (directory / f"{name}.request.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )


def refresh_persona(
    config: dict, model: Model, *, audit_dir: Path | None = None
) -> dict:
    plan = plan_refresh(config)
    processed = Path(config["paths"]["processed"])
    runtime = Path(config["paths"].get("runtime", processed.parent / "runtime"))
    audit = audit_dir or runtime / "build-audits" / "persona-refresh"
    prepare_refresh(plan, audit)
    print(
        f"PERSONA_BUILD | {plan.source['messages']} recent messages: {plan.source['first']} — {plan.source['last']}. Profile: request 1/3.",
        flush=True,
    )
    raw_profile = model.complete(plan.profile_messages)
    (audit / "profile.response.txt").write_text(raw_profile, encoding="utf-8")
    if not raw_profile.strip():
        raise ValueError(
            "Profile refresh returned empty text; existing artifacts preserved"
        )
    print(
        "PERSONA_BUILD | Profile received. Life Prompt from the same history: request 2/3.",
        flush=True,
    )
    raw_life = model.complete(plan.life_messages)
    (audit / "life.response.txt").write_text(raw_life, encoding="utf-8")
    if not raw_life.strip():
        raise ValueError(
            "Life prompt refresh returned empty text; existing artifacts preserved"
        )
    print(
        "PERSONA_BUILD | Life Prompt received. Diary from the same history: request 3/3.",
        flush=True,
    )
    raw_diary = model.complete(plan.diary_messages)
    (audit / "diary.response.txt").write_text(raw_diary, encoding="utf-8")
    entries = Diary._parse_entries(raw_diary)
    decoded = json.loads(
        raw_diary.strip()
        .removeprefix("```json")
        .removeprefix("```")
        .removesuffix("```")
        .strip()
    )
    evidence = {archive_source_id(m): m for m in plan.history.messages}
    zone = ZoneInfo(plan.source["timezone"])
    now = datetime.now(zone)
    linked: list[Entry] = []
    for entry, row in zip(entries, decoded["entries"]):
        ids = row.get("source_ids")
        if (
            not isinstance(ids, list)
            or not ids
            or not all(isinstance(key, str) and key in evidence for key in ids)
        ):
            raise ValueError(
                "Refreshed diary contains unknown source IDs; existing artifacts preserved"
            )
        if entry.timestamp.date() not in {
            evidence[key].timestamp.date() for key in ids
        }:
            raise ValueError(
                "Refreshed diary date does not match its sources; existing artifacts preserved"
            )
        linked.append(Entry(entry.timestamp, entry.content, sorted(set(ids)), now))
    # Both model outputs are validated before replacing any accepted artifact.
    # Keep the previous artifacts for inspection; do not discard older memories.
    backup = processed / "persona_backups" / now.strftime("%Y%m%d-%H%M%S-%f")
    backup.mkdir(parents=True, exist_ok=True)
    for filename in (
        config["paths"]["inner_prompt_filename"],
        config["paths"].get("life_prompt_filename", "life_prompt.md"),
        config["paths"]["diary_filename"],
        "persona_sources.json",
    ):
        old = processed / filename
        if old.exists():
            (backup / old.name).write_bytes(old.read_bytes())
    profile_path = processed / config["paths"]["inner_prompt_filename"]
    temporary_profile = profile_path.with_suffix(profile_path.suffix + ".tmp")
    InnerPrompt(raw_profile.strip()).save(temporary_profile)
    life_prompt_path = processed / config["paths"].get(
        "life_prompt_filename", "life_prompt.md"
    )
    temporary_life = life_prompt_path.with_suffix(life_prompt_path.suffix + ".tmp")
    LifePrompt(raw_life.strip()).save(temporary_life)
    diary_path = processed / config["paths"]["diary_filename"]
    temporary_diary = diary_path.with_suffix(diary_path.suffix + ".tmp")
    Diary(sorted(linked, key=lambda e: e.timestamp.isoformat())).save(temporary_diary)
    temporary_profile.replace(profile_path)
    temporary_life.replace(life_prompt_path)
    temporary_diary.replace(diary_path)
    metadata = {
        **plan.source,
        "built_at": now.isoformat(),
        "model": plan.model["model"],
    }
    save_manifest(
        processed,
        {"version": 1, "profile": metadata, "life": metadata, "diary": metadata},
    )
    print(
        f"PERSONA_READY | New profile, Life Prompt, and {len(linked)} diary entries saved. Previous versions: {backup}.",
        flush=True,
    )
    return metadata
