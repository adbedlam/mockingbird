import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest
from config_helpers import scheduler_settings

from mockingbird.chat.chat import Chat
from mockingbird.chat.event_recorder.event_recorder import PipelineEvent
from mockingbird.conversation.history import History
from mockingbird.conversation.message import Message
from mockingbird.mockingbird.mockingbird import Mockingbird
from mockingbird.persona.diary.diary import Diary
from mockingbird.persona.diary.entry import Entry
from mockingbird.persona.diary.runtime import DiaryMemory
from mockingbird.persona.inner.inner import InnerPrompt
from mockingbird.persona.refresh import (
    plan_refresh,
    prepare_refresh,
    refresh_needed,
    refresh_persona,
)
from mockingbird.persona.source import select_persona_history
from mockingbird.preprocessing.telegram.source import load_latest_history

ZONE = ZoneInfo("Europe/Moscow")
NOW = datetime(2026, 11, 17, 23, tzinfo=ZONE)


def incoming(text="I prefer short replies", seconds=0) -> Message:
    return Message("Partner", NOW + timedelta(seconds=seconds), text=text)


def memory_at(path: Path, **kwargs) -> DiaryMemory:
    return DiaryMemory(
        path,
        "Persona",
        "Partner",
        Diary([Entry(datetime(2024, 1, 1), "OLD_MEMORY_2024")]),
        **kwargs,
    )


def response_for(
    batch, content="Partner asked for concise replies", reason="Useful preference"
):
    evidence = json.loads(batch.messages[-1]["content"])["incoming_evidence"][0]
    return json.dumps(
        {
            "entries": [
                {
                    "timestamp": evidence["timestamp"],
                    "content": content,
                    "source_ids": [evidence["id"]],
                }
            ],
            "reason": reason,
        }
    )


def test_memory_ignores_generated_replies_and_deduplicates_across_restart(
    tmp_path: Path,
):
    memory = memory_at(tmp_path)
    generated = Message("Persona", NOW, text="Invented personal event")
    assert memory.ingest([incoming(), generated, incoming()]) == 1
    batch = memory.prepare(NOW)
    assert "Invented personal event" not in json.dumps(batch.messages)
    result = memory.apply(batch, response_for(batch), NOW)
    assert result.added == 1
    restored = memory_at(tmp_path)
    assert restored.ingest([incoming()]) == 0
    assert len(restored.entries) == 1
    assert restored.entries[0].source_ids == list(batch.ids)
    assert restored.sources[batch.ids[0]]["text"] == incoming().text
    assert not restored.pending


def test_pending_below_threshold_survives_restart_and_manual_flush_is_possible(
    tmp_path: Path,
):
    memory = memory_at(tmp_path, update_every=200)
    memory.ingest([incoming()])
    assert not memory.due(NOW)
    restored = memory_at(tmp_path)
    assert len(restored.pending) == 1
    assert restored.due(NOW, force=True)


def test_failed_persistence_does_not_consume_memory_batch(tmp_path: Path, monkeypatch):
    memory = memory_at(tmp_path)
    memory.ingest([incoming()])
    batch = memory.prepare(NOW)

    def fail():
        raise OSError("disk failure")

    monkeypatch.setattr(memory, "save", fail)
    with pytest.raises(OSError):
        memory.apply(batch, response_for(batch), NOW)
    assert not memory.entries
    assert not memory.processed
    assert len(memory.pending) == 1


def test_skip_commits_seen_ids_and_visible_reason(tmp_path: Path):
    memory = memory_at(tmp_path)
    memory.ingest([incoming("hi")])
    batch = memory.prepare(NOW)
    result = memory.apply(
        batch, json.dumps({"entries": [], "reason": "No durable information"}), NOW
    )
    assert result.added == 0
    assert memory.ingest([incoming("hi")]) == 0
    assert "No durable information" in (tmp_path / "diary_updates.md").read_text(
        encoding="utf-8"
    )
    assert "OLD_MEMORY_2024" in (tmp_path / "diary.md").read_text(encoding="utf-8")


@pytest.mark.parametrize("corruption", ["source", "date", "content", "json"])
def test_invalid_memory_cannot_consume_pending_evidence(
    tmp_path: Path, corruption: str
):
    memory = memory_at(tmp_path)
    memory.ingest([incoming()])
    batch = memory.prepare(NOW)
    data = json.loads(response_for(batch))
    if corruption == "source":
        data["entries"][0]["source_ids"] = ["made-up-id"]
    elif corruption == "date":
        data["entries"][0]["timestamp"] = "2030-01-01T12:00:00+03:00"
    elif corruption == "content":
        data["entries"][0]["content"] = ""
    with pytest.raises(ValueError):
        memory.apply(
            batch, "invalid" if corruption == "json" else json.dumps(data), NOW
        )
    assert len(memory.pending) == 1
    assert not memory.entries
    assert not memory.processed


def test_new_incoming_during_update_is_preserved_for_next_batch(tmp_path: Path):
    memory = memory_at(tmp_path)
    memory.ingest([incoming()])
    batch = memory.prepare(NOW)
    memory.ingest([incoming("A newer message", seconds=5)])
    memory.apply(batch, response_for(batch), NOW + timedelta(seconds=10))
    assert [e["text"] for e in memory.pending] == ["A newer message"]
    with pytest.raises(ValueError, match="already been processed"):
        memory.apply(batch, response_for(batch), NOW + timedelta(seconds=10))


def test_exact_duplicate_diary_content_is_not_appended_twice(tmp_path: Path):
    memory = memory_at(tmp_path)
    for offset in (0, 5):
        memory.ingest([incoming(seconds=offset)])
        batch = memory.prepare(NOW + timedelta(seconds=offset))
        memory.apply(
            batch, response_for(batch, "Same memory"), NOW + timedelta(seconds=offset)
        )
    assert len(memory.entries) == 1
    assert len(memory.processed) == 2


@pytest.fixture
def config(tmp_path: Path) -> dict:
    processed = tmp_path / "processed"
    History(
        "Persona",
        "Partner",
        [
            Message("Persona", datetime(2024, 1, 1), text="OLD_MESSAGE_2024"),
            Message("Partner", datetime(2026, 6, 1, 8), text="recent context"),
            Message("Persona", datetime(2026, 6, 1, 9), text="CURRENT_STYLE_2026"),
            Message("Persona", datetime(2026, 6, 2, 9), text="LATEST_STYLE_2026"),
        ],
    ).save(processed / "history.json")
    InnerPrompt("OLD_PROFILE_2024").save(processed / "inner.md")
    Diary([Entry(datetime(2024, 1, 1), "OLD_MEMORY_2024")]).save(
        processed / "diary.json"
    )
    return {
        "user": "Persona",
        "other": "Partner",
        "preprocessing": {"latest_messages": 3},
        "persona_build": {"recent_messages": 2},
        "building_model": {
            "host": "https://example.invalid/v1",
            "model": "fake",
            "temperature": 0.3,
        },
        "inference_model": {"host": "https://example.invalid/v1", "model": "fake"},
        "scheduler": scheduler_settings(
            reply_mode="immediate",
            max_initiations_per_day=0,
            min_reply_delay_seconds=1,
        ),
        "paths": {
            "raw": tmp_path / "raw",
            "processed": processed,
            "history_filename": "history.json",
            "inner_prompt_filename": "inner.md",
            "diary_filename": "diary.json",
        },
    }


def test_latest_source_and_profile_sample_use_tail_not_fixed_2024_slice(config: dict):
    history, origin = load_latest_history(config)
    assert len(history.messages) == 3
    selected = select_persona_history(history, config)
    assert [m.text for m in selected.messages] == [
        "CURRENT_STYLE_2026",
        "LATEST_STYLE_2026",
    ]
    assert "history.json" in origin


def test_new_raw_export_is_used_and_old_raw_cannot_downgrade_cache(config: dict):
    raw = config["paths"]["raw"]
    raw.mkdir()
    for year, expected in ((2024, "history.json"), (2026, "result.json")):
        (raw / "result.json").write_text(
            json.dumps(
                {
                    "messages": [
                        {
                            "id": 1,
                            "type": "message",
                            "from": "Persona",
                            "date": f"{year}-11-01T12:00:00",
                            "text": "raw message",
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        history, origin = load_latest_history(config)
        assert expected in origin
        assert max(m.timestamp.year for m in history.messages) == 2026


class RefreshModel:
    def __init__(self, invalid=False):
        self.calls = 0
        self.invalid = invalid

    def complete(self, messages: list) -> str:
        self.calls += 1
        if self.calls == 1:
            return "NEW_PROFILE_2026"
        if self.calls == 2:
            return "NEW_LIFE_PROMPT_2026"
        evidence = json.loads(messages[-1]["content"])[-1]
        return json.dumps(
            {
                "entries": [
                    {
                        "timestamp": evidence["timestamp"],
                        "content": "New grounded memory",
                        "source_ids": ["unknown" if self.invalid else evidence["id"]],
                    }
                ]
            }
        )


def test_refresh_is_preparable_without_model_and_uses_consistent_sources(
    config: dict, tmp_path: Path
):
    plan = plan_refresh(config)
    prepare_refresh(plan, tmp_path / "review")
    profile = json.loads(
        (tmp_path / "review/profile.request.json").read_text(encoding="utf-8")
    )
    diary = json.loads(
        (tmp_path / "review/diary.request.json").read_text(encoding="utf-8")
    )
    life = json.loads(
        (tmp_path / "review/life.request.json").read_text(encoding="utf-8")
    )
    assert profile["source"] == diary["source"] == life["source"]
    assert profile["source"]["first"].startswith("2026")
    assert "OLD_MESSAGE_2024" not in json.dumps(profile)
    assert "api_key" not in json.dumps(profile)


def test_refresh_validates_both_outputs_and_backs_up_old_artifacts(config: dict):
    processed = config["paths"]["processed"]
    before = (processed / "diary.json").read_bytes()
    model = RefreshModel()
    metadata = refresh_persona(config, model)
    assert model.calls == 3
    assert metadata["last"].startswith("2026")
    assert (processed / "inner.md").read_text(encoding="utf-8") == "NEW_PROFILE_2026"
    assert (processed / "life_prompt.md").read_text(
        encoding="utf-8"
    ) == "NEW_LIFE_PROMPT_2026"
    assert Diary.load(processed / "diary.json").entries[0].source_ids
    assert (
        list((processed / "persona_backups").rglob("diary.json"))[0].read_bytes()
        == before
    )
    manifest = json.loads(
        (processed / "persona_sources.json").read_text(encoding="utf-8")
    )
    assert manifest["profile"]["sha256"] == manifest["diary"]["sha256"]
    assert manifest["life"]["sha256"] == manifest["profile"]["sha256"]
    assert not refresh_needed(config)
    (processed / "inner.md").unlink()
    assert refresh_needed(config)


def test_failed_diary_refresh_preserves_both_accepted_artifacts(config: dict):
    processed = config["paths"]["processed"]
    before = {
        name: (processed / name).read_bytes() for name in ("inner.md", "diary.json")
    }
    with pytest.raises(ValueError, match="unknown source IDs"):
        refresh_persona(config, RefreshModel(invalid=True))
    assert before == {name: (processed / name).read_bytes() for name in before}


class RuntimeModel:
    def complete(self, messages: list) -> str:
        try:
            evidence = json.loads(messages[-1]["content"])["incoming_evidence"][0]
        except (ValueError, KeyError, TypeError):
            return json.dumps(
                {
                    "send": True,
                    "text": "GENERATED_REPLY_NOT_A_FACT",
                    "reason": "Answering",
                }
            )
        return json.dumps(
            {
                "entries": [
                    {
                        "timestamp": evidence["timestamp"],
                        "content": "Partner told me their preference",
                        "source_ids": [evidence["id"]],
                    }
                ],
                "reason": "Durable preference",
            }
        )


def test_current_context_excludes_old_memories_but_keeps_new_cross_session_evidence(
    config: dict,
):
    model = RuntimeModel()
    mockingbird = Mockingbird.from_config(config, model=model)
    assert mockingbird.memory is not None
    mockingbird.memory.ingest([incoming("NEW_PREFERENCE_2026")])
    decision = mockingbird.scheduler.tick(NOW)
    payload = json.dumps(mockingbird.context_builder.build(decision, NOW))
    assert "OLD_MEMORY_2024" not in payload
    assert "OLD_MESSAGE_2024" not in payload
    assert "NEW_PREFERENCE_2026" in payload
    assert "CURRENT_STYLE_2026" in payload
    assert "OLD_MEMORY_2024" in str(mockingbird.memory.base)


@pytest.mark.parametrize("manual", [False, True])
def test_pipeline_updates_diary_after_reply_with_visible_sources(
    config: dict, tmp_path: Path, manual: bool
):
    config["diary"] = {"update_in": 200 if manual else 1}
    mockingbird = Mockingbird.from_config(config, model=RuntimeModel())
    now = [NOW]
    events: list[PipelineEvent] = []
    pipeline = Chat(
        mockingbird,
        tmp_path / "session",
        events.append,
        clock=lambda: now[0],
        max_generations=2,
    )
    pipeline.start()
    pipeline.receive("NEW_PREFERENCE_2026")
    if manual:
        pipeline.command("/diary update")
    now[0] += timedelta(seconds=31)
    deadline = time.monotonic() + 3
    while (
        not any(e.stage == "DIARY_ADDED" for e in events)
        and time.monotonic() < deadline
    ):
        pipeline.step()
        time.sleep(0.005)
    pipeline.close()
    assert any(e.stage == "OUTPUT" for e in events)
    assert any(e.stage == "DIARY_ADDED" for e in events)
    assert pipeline.generations == 2
    assert mockingbird.memory is not None and len(mockingbird.memory.entries) == 1
    assert not mockingbird.memory.pending
    assert "GENERATED_REPLY_NOT_A_FACT" not in json.dumps(mockingbird.memory.sources)
    assert "Partner told me their preference" in json.dumps(
        mockingbird.context_builder.build(mockingbird.scheduler.tick(now[0]), now[0])
    )
    assert (mockingbird.memory.directory / "diary.md").exists()


def test_history_zero_limit_does_not_expand_to_whole_archive(config: dict):
    history, _ = load_latest_history(config)
    assert history.to_llm_messages("Persona", 0) == []
    with pytest.raises(ValueError):
        history.to_llm_messages("Persona", -1)


def test_automatic_refresh_respects_total_request_budget(
    config: dict, tmp_path: Path, monkeypatch, capsys
):
    from mockingbird import entrypoint
    from mockingbird.interfaces.terminal import interface

    config["persona_build"]["auto_refresh_on_start"] = True
    monkeypatch.setattr(entrypoint, "load_config", lambda path: config)

    def forbidden(*args, **kwargs):
        raise AssertionError("No refresh allowed under a one-request budget")

    monkeypatch.setattr(interface, "refresh_persona", forbidden)
    monkeypatch.setattr(
        "sys.argv",
        [
            "mockingbird",
            "--interface",
            "cli",
            "--state-dir",
            str(tmp_path / "runtime"),
            "--max-generations",
            "1",
        ],
    )
    with pytest.raises(SystemExit) as exc:
        entrypoint.main()
    assert exc.value.code == 1
    assert "three requests" in capsys.readouterr().err
