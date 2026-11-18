import json
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest
from test_memory_versions import apply, memory_at, operation
from test_pipeline import Model, bird_for, pump

from mockingbird.conversation.message import Message
from mockingbird.persona.diary.diary import Diary
from mockingbird.persona.diary.entry import Entry
from mockingbird.persona.diary.retrieval import MemorySearchConfig, select_memories
from mockingbird.pipeline import ChatPipeline

NOW = datetime(2026, 11, 18, 13, tzinfo=ZoneInfo("Europe/Moscow"))


def corpus():
    old = Entry(
        NOW - timedelta(days=900),
        "Партнёр закончил интеграцию телеграма",
        ["old-source"],
        memory_id="telegram",
        state="completed",
        subject="partner",
        topic="telegram.integration",
    )
    recent = [
        Entry(
            NOW - timedelta(days=50 - index),
            f"Обсудили маршрут поездки номер {index}",
            memory_id=f"trip{index}",
        )
        for index in range(45)
    ]
    return [old, *recent]


def select(entries, query, **kwargs):
    return select_memories(entries, query, NOW, timezone="Europe/Moscow", **kwargs)


def test_old_relevant_memory_is_found_beyond_latest_30_and_is_marked_historical():
    result = select(corpus(), "что с интеграцией телеграма", limit=6)
    match = next(hit for hit in result.hits if hit.entry.memory_id == "telegram")
    assert match.reason == "relevance" and match.historical
    assert match.entry.state == "completed" and match.score > 0
    assert match.matched_terms
    assert len(result.hits) == 6
    assert any(hit.reason == "recent" for hit in result.hits)
    assert result.chars <= result.max_chars


def test_unrelated_empty_and_disabled_search_do_not_pull_in_old_records():
    for query, config in [
        ("совсем другая тема", None),
        ("", None),
        ("телеграм", MemorySearchConfig(enabled=False)),
    ]:
        result = select(corpus(), query, limit=6, config=config)
        assert all(hit.entry.memory_id != "telegram" for hit in result.hits)
        assert len(result.hits) == 6


def test_current_query_has_priority_over_previous_conversation_topic():
    result = select_memories(
        corpus(),
        "телеграм",
        NOW,
        timezone="Europe/Moscow",
        limit=1,
        context_query="маршрут поездка маршрут поездка",
    )
    assert result.entries[0].memory_id == "telegram"
    reference = select_memories(
        corpus(),
        "а он",
        NOW,
        timezone="Europe/Moscow",
        limit=1,
        context_query="телеграм",
    )
    assert reference.entries[0].memory_id == "telegram"


def test_future_and_superseded_versions_are_not_searchable():
    entries = corpus()
    entries[0] = replace(entries[0], revision=1)
    entries.append(
        Entry(
            NOW - timedelta(days=901),
            "Планировали интеграцию телеграма",
            memory_id="telegram",
            revision=0,
            state="planned",
        )
    )
    entries.append(
        Entry(
            NOW + timedelta(days=1), "Интеграция телеграма завтра", memory_id="future"
        )
    )
    result = select(entries, "телеграм", limit=2)
    assert all(hit.entry.memory_id != "future" for hit in result.hits)
    match = next(hit for hit in result.hits if hit.entry.memory_id == "telegram")
    assert match.entry.state == "completed"
    assert sum(hit.entry.memory_id == "telegram" for hit in result.hits) == 1


def test_configured_age_cap_and_character_budget_are_respected():
    result = select(corpus(), "телеграм", config=MemorySearchConfig(max_age_days=365))
    assert all(hit.entry.memory_id != "telegram" for hit in result.hits)
    large = Entry(NOW, "телеграм " * 2000, memory_id="oversized")
    result = select(
        [*corpus(), large],
        "телеграм",
        limit=3,
        config=MemorySearchConfig(max_chars=1700),
    )
    assert result.chars <= 1700
    assert all(hit.entry.memory_id != "oversized" for hit in result.hits)
    assert result.oversized >= 1


def test_normalization_aliases_and_topic_keys_help_matching():
    entries = [
        Entry(
            NOW - timedelta(days=900),
            "Настроил скедулер Пересмешника",
            memory_id="scheduler",
            topic="scheduler.timing",
        )
    ]
    result = select(
        entries,
        "scheduler mockingbird",
        config=MemorySearchConfig(
            aliases={"скедулер": "scheduler", "пересмешник": "mockingbird"}
        ),
    )
    assert len(result.hits) == 1 and result.hits[0].score > 0
    russian = select(
        [
            Entry(
                NOW - timedelta(days=900),
                "Партнёр тестировал проект",
                memory_id="tests",
            )
        ],
        "тесты проекта",
    )
    assert russian.entries[0].memory_id == "tests"


@pytest.mark.parametrize(
    "data",
    [
        {"enabled": "yes"},
        {"recent_entries": -1},
        {"max_chars": 0},
        {"min_score": 0},
        {"min_score": float("nan")},
        {"max_age_days": -1},
        {"aliases": {"two words": "one"}},
        {"unknown_option": True},
    ],
)
def test_invalid_retrieval_configuration_fails_before_model_request(data):
    with pytest.raises(ValueError):
        MemorySearchConfig.from_dict(data)


def test_deterministic_result_and_no_mutation():
    entries = corpus()
    before = [entry.to_string() for entry in entries]
    a = select(entries, "телеграм", limit=4)
    b = select(list(reversed(entries)), "телеграм", limit=4)
    assert a.report() == b.report()
    assert [entry.to_string() for entry in entries] == before


def test_diary_update_can_find_and_close_an_old_relevant_memory(tmp_path):
    memory = memory_at(tmp_path, Diary(corpus()), context_entries=4)
    memory.ingest([Message("Partner", NOW, text="Интеграцию телеграма отменили")])
    batch = memory.prepare(NOW)
    target = next(e for e in memory.combined().entries if e.memory_id == "telegram")
    assert any(key == "telegram" for key, _, _ in batch.targets)
    result = apply(
        memory,
        batch,
        [
            operation(
                batch,
                "update",
                target,
                content="Партнёр сообщил что интеграцию телеграма отменили",
                state="cancelled",
            )
        ],
        NOW,
    )
    assert result.updated == 1
    assert (
        next(e for e in memory.combined().entries if e.memory_id == "telegram").state
        == "cancelled"
    )


def test_reply_and_request_audit_include_retrieval_but_not_fake_world_facts(tmp_path):
    bird = bird_for(Model())
    bird.diary = Diary(corpus())
    bird.diary_last_n = 4
    events = []
    clock = [NOW]
    pipeline = ChatPipeline(
        bird, tmp_path, events.append, clock=lambda: clock[0], max_generations=1
    )
    pipeline.start()
    pipeline.receive("что с интеграцией телеграма")
    clock[0] += timedelta(seconds=5)
    pump(pipeline, lambda: any(e.stage == "OUTPUT" for e in events))
    search = next(e for e in events if e.stage == "MEMORY_SEARCH")
    assert any(
        row["memory_id"] == "telegram" and row["historical"]
        for row in search.data["selection"]["selected"]
    )
    request = json.loads(pipeline.last_request.read_text(encoding="utf-8"))
    assert request["memory_selection"] == search.data["selection"]
    diary_prompt = next(
        m["content"]
        for m in request["messages"]
        if m["content"].startswith("Dated diary")
    )
    assert "[historical: outside recent window]" in diary_prompt
    assert "закончил интеграцию" in diary_prompt
    pipeline.close()


def test_memory_command_is_local_and_does_not_enter_conversation(tmp_path):
    model = Model()
    bird = bird_for(model)
    bird.diary = Diary(corpus())
    events = []
    pipeline = ChatPipeline(bird, tmp_path, events.append, clock=lambda: NOW)
    pipeline.paused = True
    pipeline.start()
    pipeline.command("/memory телеграм")
    assert not model.calls
    assert not bird.live.messages
    event = next(e for e in events if e.stage == "MEMORY_SEARCH")
    assert "закончил интеграцию" in event.data["text"]
    assert event.data["selection"]["selected"][0]["historical"]
    pipeline.close()
