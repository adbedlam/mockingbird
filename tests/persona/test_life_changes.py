import json
from copy import deepcopy
from datetime import timedelta
from pathlib import Path

import pytest
from tests.chat.test_pipeline import Clock, Model, bird_for, pump
from tests.persona.test_life import life_for, plan

from mockingbird.chat.chat import Chat
from mockingbird.mockingbird.draft import parse_draft
from mockingbird.scheduler.scheduler import MessageKind


def setup_life(tmp_path):
    now = Clock().now.replace(hour=13)
    life = life_for(tmp_path)
    life.apply(plan(), "2026-11-17", now)
    incoming = [
        {
            "id": "incoming-1",
            "author": "Partner",
            "timestamp": now.isoformat(),
            "text": "давай вместо прогулки вечером потестим проект с 18 до 19",
        }
    ]
    return life, now, incoming


def replacement(source="incoming-1", revision=0):
    return {
        "date": "2026-11-17",
        "base_revision": revision,
        "source_id": source,
        "reason": "Согласились вечером заняться проектом вместо прогулки",
        "changes": [
            {
                "op": "replace",
                "id": "a3",
                "start": "18:00",
                "end": "19:00",
                "activity": "потестировал проект",
            }
        ],
    }


def test_agreed_change_preserves_past_current_and_updates_diary_restart_context(
    tmp_path,
):
    life, now, incoming = setup_life(tmp_path)
    frozen = deepcopy(life.days["2026-11-17"]["activities"][:2])
    life.advance(now)
    change = life.update(replacement(), now, incoming, "давай вечером потестим")
    day = life.days["2026-11-17"]
    assert day["activities"][:2] == frozen
    assert day["revision"] == 1
    assert change["source"] == incoming[0]
    assert change["reply"] == "давай вечером потестим"
    assert "погулял" in life.describe_change(
        change
    )  # The canceled description remains auditable.
    assert day["activities"][2]["activity"] == "потестировал проект"
    assert '"revision": 1' in life.context(now)
    restored = life_for(tmp_path)
    assert not restored.advance(
        now
    )  # Existing completed/current events are not repeated.
    restored.advance(now.replace(hour=23))
    assert "потестировал проект" in restored.diary_view()
    assert "погулял" not in restored.diary_view()
    assert "Incoming:" in (life.directory / "plan_changes.md").read_text(
        encoding="utf-8"
    )
    with pytest.raises(ValueError, match="stale"):
        restored.update(replacement(), now, incoming, "повтор")


@pytest.mark.parametrize(
    "mutate",
    [
        lambda p: p.update(date="2026-11-18"),
        lambda p: p.update(base_revision=4),
        lambda p: p.update(base_revision=False),
        lambda p: p.update(source_id="old-or-fabricated-message"),
        lambda p: p.update(reason=""),
        lambda p: p.update(changes=[]),
        lambda p: p["changes"][0].update(id="unknown"),
        lambda p: p["changes"][0].update(id="a1"),  # completed
        lambda p: p["changes"][0].update(id="a2"),  # current
        lambda p: p["changes"][0].update(start="12:00", end="13:30"),
        lambda p: p["changes"][0].update(start="13:00"),  # exactly now
        lambda p: p["changes"][0].update(start="13:30"),  # overlaps current
        lambda p: p["changes"][0].update(end="25:00"),
        lambda p: p["changes"][0].update(activity=""),
        lambda p: p["changes"][0].update(op="invent_past"),
        lambda p: p["changes"][0].update(activity="погулял"),  # no change
        lambda p: p["changes"].append(deepcopy(p["changes"][0])),
    ],
)
def test_invalid_change_set_preserves_all_world_state_and_disk(tmp_path, mutate):
    life, now, incoming = setup_life(tmp_path)
    before = deepcopy(life.days)
    disk = (life.directory / "life.json").read_bytes()
    update = replacement()
    mutate(update)
    with pytest.raises(ValueError):
        life.update(update, now, incoming, "давай")
    assert life.days == before
    assert (life.directory / "life.json").read_bytes() == disk


def test_cancel_add_and_failed_second_operation_are_atomic(tmp_path):
    life, now, incoming = setup_life(tmp_path)
    update = replacement()
    update["changes"] = [
        {"op": "cancel", "id": "a3"},
        {
            "op": "add",
            "start": "18:00",
            "end": "19:00",
            "activity": "потестировал проект",
        },
    ]
    life.update(update, now, incoming, "давай")
    day = life.days["2026-11-17"]
    assert [r["id"] for r in day["activities"]] == ["a1", "a2", "a4"]
    before = deepcopy(day)
    update["base_revision"] = 1
    update["changes"] = [
        {"op": "cancel", "id": "a4"},
        {"op": "add", "start": "09:00", "end": "10:00", "activity": "retroactive"},
    ]
    with pytest.raises(ValueError):
        life.update(update, now, incoming, "давай")
    assert life.days["2026-11-17"] == before


def test_cancel_all_future_activities_can_leave_an_empty_day(tmp_path):
    life, now, incoming = setup_life(tmp_path)
    earlier = now.replace(hour=8)
    update = replacement()
    update["changes"] = [{"op": "cancel", "id": f"a{i}"} for i in (1, 2, 3)]
    life.update(update, earlier, incoming, "давай сегодня без дел")
    restored = life_for(tmp_path)
    assert restored.days["2026-11-17"]["activities"] == []
    assert not restored.needed(earlier)
    restored.advance(now.replace(hour=23))
    assert "no completed activities yet" in restored.diary_view()


def test_old_life_state_migrates_ids_and_event_markers_without_repeating_events(
    tmp_path,
):
    life, now, _ = setup_life(tmp_path)
    state = json.loads((life.directory / "life.json").read_text(encoding="utf-8"))
    state["version"] = 1
    day = state["days"]["2026-11-17"]
    for row in day["activities"]:
        row.pop("id")
    for key in ("revision", "changes", "next_activity_id"):
        day.pop(key)
    day["announced"] = ["0:completed", "1:current"]
    (life.directory / "life.json").write_text(json.dumps(state), encoding="utf-8")
    restored = life_for(tmp_path)
    assert not restored.advance(now)
    assert restored.days["2026-11-17"]["revision"] == 0
    assert restored.days["2026-11-17"]["announced"] == ["a1:completed", "a2:current"]


def pipeline_for(tmp_path, model, max_generations=2):
    life, now, _ = setup_life(tmp_path)
    clock = Clock()
    clock.now = now
    mockingbird = bird_for(model)
    mockingbird.life = life
    events = []
    pipeline = Chat(
        mockingbird,
        tmp_path / "runtime",
        events.append,
        clock=clock,
        max_generations=max_generations,
    )
    pipeline.start()
    return pipeline, clock, events


def change_reply(pipeline):
    return json.dumps(
        {
            "send": True,
            "text": "давай вечером потестим проект",
            "reason": "соглашаюсь",
            "life_update": replacement(
                pipeline.mockingbird.memory_query.pending_incoming()[-1]["id"]
            ),
        }
    )


def test_pipeline_change_and_followup_context_use_one_reply_call_each(tmp_path):
    model = Model()
    pipeline, clock, events = pipeline_for(tmp_path, model)
    pipeline.receive("давай вместо прогулки вечером потестим проект с 18 до 19")
    model.responses = [
        change_reply(pipeline),
        json.dumps(
            {
                "send": True,
                "text": "вечером проект потестим",
                "reason": "помню договорённость",
                "life_update": None,
            }
        ),
    ]
    clock.now += timedelta(seconds=5)
    pump(pipeline, lambda: any(e.stage == "LIFE_CHANGED" for e in events))
    assert len(model.calls) == 1
    assert pipeline.mockingbird.life.days["2026-11-17"]["revision"] == 1
    assert (
        pipeline.mockingbird.live.messages[-1].text == "давай вечером потестим проект"
    )
    pipeline.receive("что вечером делать будешь")
    clock.now += timedelta(seconds=5)
    pump(pipeline, lambda: sum(e.stage == "OUTPUT" for e in events) == 2)
    state = next(
        m["content"]
        for m in model.calls[-1]
        if m["content"].startswith("PERMITTED DAY SIMULATION")
    )
    assert '"revision": 1' in state
    assert "потестировал проект" in state
    assert sum(e.stage == "LIFE_CHANGED" for e in events) == 1
    requests = [
        json.loads(p.read_text(encoding="utf-8"))
        for p in (pipeline.directory / "requests").glob("*.json")
    ]
    assert sum(bool(r.get("life_change")) for r in requests) == 1
    clock.now = clock.now.replace(hour=23)
    pipeline.close()
    assert "потестировал проект" in pipeline.mockingbird.life.diary_view()
    assert "погулял" not in pipeline.mockingbird.life.diary_view()


@pytest.mark.parametrize(
    "discard", ["new_input", "pause", "started_during_generation", "day_changed"]
)
def test_discarded_or_late_reply_never_changes_plan(tmp_path, discard):
    model = Model(block=True)
    pipeline, clock, events = pipeline_for(tmp_path, model, max_generations=1)
    pipeline.receive("давай вместо прогулки проект с 18 до 19")
    model.responses = [change_reply(pipeline)]
    clock.now += timedelta(seconds=5)
    pipeline.step()
    assert model.started.wait(1)
    if discard == "new_input":
        pipeline.receive("передумал прогулку оставим")
    elif discard == "pause":
        pipeline.command("/pause")
    elif discard == "day_changed":
        clock.now += timedelta(days=1)
    else:
        clock.now = clock.now.replace(hour=18, minute=1)
    model.release.set()
    pump(pipeline, lambda: pipeline.inflight is None)
    assert not any(e.stage in ("LIFE_CHANGED", "OUTPUT") for e in events)
    assert pipeline.mockingbird.life.days["2026-11-17"]["revision"] == 0
    pipeline.close()


def test_failed_plan_save_keeps_reply_pending_and_no_acceptance_output(
    tmp_path, monkeypatch
):
    model = Model()
    pipeline, clock, events = pipeline_for(tmp_path, model, max_generations=1)
    pipeline.receive("давай вместо прогулки проект с 18 до 19")
    model.responses = [change_reply(pipeline)]
    original = Path.replace

    def fail(source, target):
        if source.name == "life.tmp":
            raise OSError("simulated disk error")
        return original(source, target)

    monkeypatch.setattr(Path, "replace", fail)
    clock.now += timedelta(seconds=5)
    pump(pipeline, lambda: any(e.stage == "ERROR" for e in events))
    assert pipeline.mockingbird.scheduler.reply_due is not None
    assert pipeline.mockingbird.scheduler.last_outgoing is None
    assert len(pipeline.mockingbird.live.messages) == 1
    assert pipeline.mockingbird.life.days["2026-11-17"]["revision"] == 0
    assert not any(e.stage == "LIFE_CHANGED" for e in events)


def test_plain_reply_and_skipped_initiative_cannot_implicitly_change_life(tmp_path):
    life, now, _ = setup_life(tmp_path)
    draft = parse_draft("давай вечером поиграем", MessageKind.REPLY)
    assert draft.life_update is None
    with pytest.raises(ValueError, match="Only a sent reply"):
        parse_draft(
            json.dumps(
                {
                    "send": False,
                    "text": "",
                    "reason": "skip",
                    "life_update": replacement(),
                }
            ),
            MessageKind.INITIATIVE,
        )
    assert life.days["2026-11-17"]["revision"] == 0
