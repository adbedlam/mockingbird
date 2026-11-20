import json
from datetime import datetime, timedelta
from pathlib import Path

import pytest
from test_pipeline import Clock, Model, bird_for, pump

from mockingbird.chat.pipeline import ChatPipeline
from mockingbird.conversation.history import History
from mockingbird.conversation.message import Message as ConversationMessage
from mockingbird.persona.life import LifePrompt, PersonaLife


def plan(day="2026-11-17"):
    return json.dumps(
        {
            "date": day,
            "activities": [
                {
                    "start": "09:00",
                    "end": "10:00",
                    "activity": "разобрал файлы проекта",
                },
                {"start": "12:00", "end": "14:00", "activity": "поправил интерфейс"},
                {"start": "18:00", "end": "19:00", "activity": "погулял"},
            ],
            "reason": "Выбрал обычные дела по интересам персоны",
        },
        ensure_ascii=False,
    )


def life_for(tmp_path):
    return PersonaLife(tmp_path / "life", "Persona", "Partner", "Europe/Moscow")


def test_life_prompt_is_built_from_history_and_used_by_plan_request(tmp_path):
    class BuildingModel:
        def __init__(self):
            self.calls = []

        def complete(self, messages):
            self.calls.append(messages)
            return "Grounded Life Prompt: use recurring project and walking interests."

    history = History(
        "Persona",
        "Partner",
        [
            ConversationMessage(
                "Persona",
                datetime(2026, 11, 17, 10),
                "I have been improving my project",
            )
        ],
    )
    model = BuildingModel()
    prompt = LifePrompt.from_history(model, history)

    assert "improving my project" in model.calls[0][1]["content"]
    assert "simulated day" in model.calls[0][0]["content"]

    life = life_for(tmp_path)
    life.life_prompt = prompt.content
    messages = life.prepare(Clock().now, "INNER PROFILE")
    assert "Grounded Life Prompt" in messages[0]["content"]
    assert "INNER PROFILE" in messages[0]["content"]


def test_life_plan_has_completed_current_and_future_and_separate_origin(tmp_path):
    life = life_for(tmp_path)
    now = Clock().now.replace(hour=13)
    life.apply(plan(), "2026-11-17", now)
    rows = life.rows(life.days["2026-11-17"], now)
    assert [r["status"] for r in rows] == ["completed", "current", "planned"]
    assert life.days["2026-11-17"]["origin"] == "simulated"
    assert "fictional" in life.context(now)
    assert not life.needed(now)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda p: p.update(date="2026-11-16"),
        lambda p: p.update(activities=[]),
        lambda p: p["activities"][1].update(start="09:30"),
        lambda p: p["activities"][1].update(end="25:00"),
        lambda p: p["activities"][0].update(activity=""),
        lambda p: p.update(reason=""),
    ],
)
def test_invalid_plan_never_enters_state(tmp_path, mutation):
    life = life_for(tmp_path)
    data = json.loads(plan())
    mutation(data)
    with pytest.raises(ValueError):
        life.apply(json.dumps(data), "2026-11-17", Clock().now)
    assert life.days == {}
    assert not (life.directory / "life.json").exists()


def test_once_per_day_diary_catchup_restart_and_future_excluded(tmp_path):
    life = life_for(tmp_path)
    now = Clock().now.replace(hour=13)
    life.apply(plan(), "2026-11-17", now)
    assert not any(stage == "DAILY_DIARY" for stage, _ in life.advance(now))
    life.advance(now, checkpoint=True)
    assert "разобрал файлы проекта" in life.diary_view()
    assert "погулял" not in life.diary_view()
    assert "поправил интерфейс" not in life.diary_view()
    restored = life_for(tmp_path)
    assert not restored.advance(now, checkpoint=True)
    tomorrow = now + timedelta(days=1)
    events = restored.advance(tomorrow)
    assert sum(stage == "DAILY_DIARY" for stage, _ in events) == 1
    assert "погулял" in restored.diary_view()
    assert len(restored.days) == 1
    assert not restored.advance(tomorrow)
    assert restored.needed(tomorrow)
    assert "recent_simulated_diary" in restored.context(tomorrow)


def test_daily_cutoff_and_duplicate_plan_rejected(tmp_path):
    life = life_for(tmp_path)
    now = Clock().now
    life.apply(plan(), "2026-11-17", now)
    assert any(stage == "DAILY_DIARY" for stage, _ in life.advance(now))
    assert not life.advance(now)
    with pytest.raises(ValueError, match="already exists"):
        life.apply(plan(), "2026-11-17", now)


def test_state_is_bound_to_persona_and_timezone(tmp_path):
    life = life_for(tmp_path)
    life.apply(plan(), "2026-11-17", Clock().now)
    with pytest.raises(ValueError, match="another persona"):
        PersonaLife(life.directory, "Partner", "Persona", "Europe/Moscow")
    with pytest.raises(ValueError, match="timezone"):
        PersonaLife(life.directory, "Persona", "Partner", "UTC")


def test_life_api_error_backoff_and_request_limit(tmp_path):
    model = Model(["{broken"])
    clock = Clock()
    bird = bird_for(model)
    bird.life = life_for(tmp_path)
    events = []
    pipeline = ChatPipeline(
        bird, tmp_path / "runtime", events.append, clock=clock, max_generations=1
    )
    pipeline.start()
    pump(pipeline, lambda: any(e.stage == "LIFE_ERROR" for e in events))
    assert bird.life.days == {}
    assert not bird.life.needed(clock.now)
    clock.now += timedelta(minutes=5)
    pipeline.step()
    assert pipeline.paused
    assert len(model.calls) == 1
    pipeline.close()


def test_pipeline_plans_before_reply_input_does_not_invalidate_plan_and_logs_diary(
    tmp_path,
):
    reply = json.dumps(
        {"send": True, "text": "разбирал проект", "reason": "учёл завершённые дела"}
    )
    model = Model([plan(), reply], block=True)
    clock = Clock()
    clock.now = clock.now.replace(hour=13)
    bird = bird_for(model)
    bird.life = life_for(tmp_path)
    events = []
    pipeline = ChatPipeline(bird, tmp_path / "runtime", events.append, clock=clock)
    pipeline.start()
    assert model.started.wait(1)
    pipeline.receive("что делал весь день")
    clock.now += timedelta(seconds=5)
    model.release.set()
    pump(pipeline, lambda: any(e.stage == "OUTPUT" for e in events))
    assert len(model.calls) == 2
    assert not any(e.stage in ("DISCARD", "SUPERSEDED") for e in events)
    state = next(
        m["content"]
        for m in model.calls[1]
        if "PERMITTED DAY SIMULATION" in m["content"]
    )
    assert "разобрал файлы проекта" in state
    assert '"status": "planned"' in state
    assert "rough typing" in "\n".join(m["content"] for m in model.calls[1])
    assert "speaking AS Persona TO Partner" in "\n".join(
        m["content"] for m in model.calls[1]
    )
    assert "БЕЗ запятых и точек" in model.calls[1][-1]["content"]
    pipeline.command("/life")
    pipeline.command("/diary")
    assert any(e.stage == "LIFE_STATE" for e in events)
    assert any(e.stage == "DAILY_DIARY" for e in events)
    pipeline.close()
    assert (bird.life.directory / "daily_diary.md").exists()
    assert not (bird.life.directory.parent / "memory.json").exists()
    paths = sorted((pipeline.directory / "requests").glob("*.json"))
    requests = [json.loads(p.read_text(encoding="utf-8")) for p in paths]
    assert {r["status"] for r in requests} == {"life_planned", "sent"}


def test_day_rollover_during_plan_discards_old_result(tmp_path):
    model = Model([plan()], block=True)
    clock = Clock()
    bird = bird_for(model)
    bird.life = life_for(tmp_path)
    pipeline = ChatPipeline(
        bird, tmp_path / "runtime", lambda e: None, clock=clock, max_generations=1
    )
    pipeline.start()
    assert model.started.wait(1)
    clock.now += timedelta(days=1)
    model.release.set()
    pump(pipeline, lambda: pipeline.inflight is None)
    assert not bird.life.days
    assert pipeline.paused
    pipeline.close()


def test_plan_atomic_failure_does_not_accept_state(tmp_path, monkeypatch):
    life = life_for(tmp_path)
    original = Path.replace

    def fail(source, target):
        if source.name == "life.tmp":
            raise OSError("simulated disk error")
        return original(source, target)

    monkeypatch.setattr(Path, "replace", fail)
    with pytest.raises(OSError):
        life.apply(plan(), "2026-11-17", Clock().now)
    assert not life.days


def test_pause_discards_plan_and_no_generation_after_limit(tmp_path):
    model = Model([plan()], block=True)
    bird = bird_for(model)
    bird.life = life_for(tmp_path)
    pipeline = ChatPipeline(
        bird, tmp_path / "runtime", lambda e: None, clock=Clock(), max_generations=1
    )
    pipeline.start()
    assert model.started.wait(1)
    pipeline.command("/pause")
    model.release.set()
    pump(pipeline, lambda: pipeline.inflight is None)
    assert not bird.life.days
    pipeline.command("/resume")
    assert pipeline.paused
    assert len(model.calls) == 1
    pipeline.close()
