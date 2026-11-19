import json
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest
from config_helpers import scheduler_settings

from mockingbird.chat.events import PipelineEvent
from mockingbird.chat.pipeline import ChatPipeline
from mockingbird.conversation.history import History
from mockingbird.conversation.message import Message
from mockingbird.interfaces.terminal import SessionLock
from mockingbird.mockingbird.mockingbird import Mockingbird
from mockingbird.persona.diary.diary import Diary
from mockingbird.persona.diary.entry import Entry
from mockingbird.persona.inner.inner import InnerPrompt
from mockingbird.scheduler.profile import BehaviorProfile
from mockingbird.scheduler.scheduler import Scheduler

ZONE = ZoneInfo("Europe/Moscow")


class Clock:
    def __init__(self):
        self.now = datetime(2026, 11, 17, 23, tzinfo=ZONE)

    def __call__(self) -> datetime:
        return self.now


class Model:
    def __init__(self, responses: list[str] | None = None, block: bool = False):
        self.responses = responses or [
            json.dumps(
                {
                    "send": True,
                    "text": "Привет \U0001f426",
                    "reason": "Отвечаю на вопрос",
                }
            )
        ]
        self.calls: list[list] = []
        self.started = threading.Event()
        self.release = threading.Event()
        if not block:
            self.release.set()

    def complete(self, messages: list) -> str:
        index = len(self.calls)
        self.calls.append(messages)
        self.started.set()
        if not self.release.wait(5):
            raise TimeoutError("test gate timed out")
        return self.responses[min(index, len(self.responses) - 1)]


def bird_for(model: Model) -> Mockingbird:
    source = History(
        "Persona",
        "Partner",
        [
            Message(
                "Partner", datetime(2026, 10, 1, 8, tzinfo=ZONE), text="archive context"
            ),
            Message(
                "Persona", datetime(2026, 10, 1, 9, tzinfo=ZONE), text="archive style"
            ),
            Message(
                "Persona",
                datetime(2026, 10, 1, 17, tzinfo=ZONE),
                text="archive style 2",
            ),
        ],
    )
    profile = BehaviorProfile(source, scheduler_settings(reply_mode="immediate"))
    return Mockingbird(
        source,
        InnerPrompt("Synthetic personality"),
        Diary([Entry(source.messages[0].timestamp, "Synthetic memory")]),
        Scheduler(profile, 7),
        model,
    )


def make_pipeline(
    tmp_path: Path, model: Model, **kwargs
) -> tuple[ChatPipeline, Clock, list[PipelineEvent]]:
    clock = Clock()
    events: list[PipelineEvent] = []
    pipeline = ChatPipeline(
        bird_for(model), tmp_path, events.append, clock=clock, **kwargs
    )
    pipeline.start()
    return pipeline, clock, events


def pump(pipeline: ChatPipeline, predicate) -> None:
    deadline = time.monotonic() + 3
    while not predicate() and time.monotonic() < deadline:
        pipeline.step()
        time.sleep(0.005)
    assert predicate(), pipeline.status()


def test_fast_reply_inactive_hours_has_full_trace_and_exact_context(tmp_path: Path):
    model = Model()
    pipeline, clock, events = make_pipeline(tmp_path, model, max_generations=1)
    pipeline.receive("Привет, как дела? \U0001f426")
    clock.now += timedelta(seconds=5)
    pump(pipeline, lambda: any(e.stage == "OUTPUT" for e in events))
    pipeline.close()
    stages = [e.stage for e in events]
    assert (
        stages.index("INPUT")
        < stages.index("SCHEDULE")
        < stages.index("REQUEST")
        < stages.index("OUTPUT")
    )
    assert len(model.calls) == 1
    assert pipeline.bird.live.messages[-1].text == "Привет \U0001f426"
    assert pipeline.bird.scheduler.initiations_sent == 0
    assert pipeline.last_request is not None
    request = json.loads(pipeline.last_request.read_text(encoding="utf-8"))
    assert request["messages"] == model.calls[0]
    assert request["status"] == "sent"
    assert request["response_raw"]
    serialized = json.dumps(request, ensure_ascii=False)
    assert "Synthetic personality" in serialized
    assert "Synthetic memory" in serialized
    assert "Привет, как дела? \U0001f426" in serialized
    transcript = (tmp_path / "chat_transcript.md").read_text(encoding="utf-8")
    assert "Привет \U0001f426" in transcript
    assert "Решение модели: Отвечаю на вопрос" in transcript


def test_incoming_and_status_work_during_generation_and_stale_reply_is_discarded(
    tmp_path: Path,
):
    old = json.dumps({"send": True, "text": "old reply", "reason": "old"})
    new = json.dumps({"send": True, "text": "new reply", "reason": "updated"})
    model = Model([old, new], block=True)
    pipeline, clock, events = make_pipeline(tmp_path, model, max_generations=2)
    pipeline.receive("first question")
    clock.now += timedelta(seconds=5)
    pipeline.step()
    assert model.started.wait(1)
    pipeline.command("/status")
    assert "выполняется" in events[-1].message
    clock.now += timedelta(seconds=1)
    pipeline.receive("second question")
    assert events[-1].stage == "SUPERSEDED"
    clock.now += timedelta(seconds=5)
    model.release.set()
    pump(pipeline, lambda: any(e.stage == "OUTPUT" for e in events))
    pipeline.close()
    assert any(e.stage == "DISCARD" for e in events)
    assert [m.text for m in pipeline.bird.live.messages] == [
        "first question",
        "second question",
        "new reply",
    ]
    assert "second question" in json.dumps(model.calls[-1])


def test_pause_invalidates_inflight_even_if_resumed_before_completion(tmp_path: Path):
    model = Model(block=True)
    pipeline, clock, events = make_pipeline(tmp_path, model, max_generations=1)
    pipeline.receive("hello")
    clock.now += timedelta(seconds=5)
    pipeline.step()
    assert model.started.wait(1)
    pipeline.command("/pause")
    pipeline.command("/resume")
    model.release.set()
    pump(pipeline, lambda: pipeline.inflight is None)
    pipeline.close()
    assert any(e.stage == "DISCARD" for e in events)
    assert not any(e.stage == "OUTPUT" for e in events)
    assert len(model.calls) == 1


def test_three_invalid_outputs_pause_retries_and_preserve_incoming(tmp_path: Path):
    model = Model(['{"send": true,'])
    pipeline, clock, events = make_pipeline(tmp_path, model)
    pipeline.receive("question")
    for attempt in range(3):
        clock.now = pipeline.bird.scheduler.reply_due
        pipeline.step()
        pump(pipeline, lambda: pipeline.failures == attempt + 1)
    for _ in range(5):
        clock.now += timedelta(hours=1)
        pipeline.step()
    pipeline.close()
    assert pipeline.paused
    assert len(model.calls) == 3
    assert pipeline.bird.live.messages[-1].text == "question"
    assert pipeline.bird.scheduler.initiations_sent == 0
    assert sum(e.stage == "ERROR" for e in events) == 3


def test_pending_reply_survives_restart_and_commands_are_not_model_input(
    tmp_path: Path,
):
    pipeline, clock, events = make_pipeline(tmp_path, Model())
    pipeline.receive("remember this")
    pipeline.command("/status")
    pipeline.command("/history")
    pipeline.command("/plan")
    pipeline.command("/context")
    assert pipeline.command("/quit") is False
    pipeline.close()
    new_model = Model()
    restored = bird_for(new_model)
    restored.restore(tmp_path)
    assert [m.text for m in restored.live.messages] == ["remember this"]
    due = restored.scheduler.reply_due
    new_events: list[PipelineEvent] = []
    next_pipeline = ChatPipeline(restored, tmp_path, new_events.append, clock=clock)
    clock.now = due
    next_pipeline.start()
    pump(next_pipeline, lambda: any(e.stage == "OUTPUT" for e in new_events))
    next_pipeline.close()
    assert len(new_model.calls) == 1
    assert "/status" not in json.dumps(new_model.calls)


def test_autonomous_slot_runs_without_any_incoming(tmp_path: Path):
    model = Model()
    bird = bird_for(model)
    clock = Clock()
    clock.now = clock.now.replace(hour=9)
    bird.scheduler.day = clock.now.date()
    bird.scheduler.planned = [clock.now]
    events: list[PipelineEvent] = []
    pipeline = ChatPipeline(
        bird, tmp_path, events.append, clock=clock, max_generations=1
    )
    pipeline.start()
    pump(pipeline, lambda: any(e.stage == "OUTPUT" for e in events))
    pipeline.close()
    assert bird.scheduler.initiations_sent == 1
    assert bird.scheduler.awaiting_reply
    assert not any(e.stage == "INPUT" for e in events)


def test_exclusive_session_lock_releases_after_exit(tmp_path: Path):
    with SessionLock(tmp_path):
        with pytest.raises(RuntimeError, match="другим процессом"):
            with SessionLock(tmp_path):
                pass
    with SessionLock(tmp_path):
        pass


def test_old_scheduler_snapshot_defaults_to_history_reply_mode(tmp_path: Path):
    bird = bird_for(Model())
    history_config = scheduler_settings()
    profile = BehaviorProfile(bird.history, history_config)
    state = tmp_path / "scheduler.json"
    Scheduler(profile).save(state, ("Persona", "Partner"))
    data = json.loads(state.read_text(encoding="utf-8"))
    del data["config"]["reply_mode"]
    state.write_text(json.dumps(data), encoding="utf-8")
    restored = Scheduler.load(state, profile, ("Persona", "Partner"))
    assert restored.config["reply_mode"] == "history"


def test_plain_reply_is_output_once_with_explicit_format_fallback(tmp_path: Path):
    model = Model(["Ответ готов обычным текстом"])
    pipeline, clock, events = make_pipeline(tmp_path, model, max_generations=1)
    pipeline.receive("question")
    clock.now += timedelta(seconds=5)
    pump(pipeline, lambda: any(e.stage == "OUTPUT" for e in events))
    pipeline.close()
    assert len(model.calls) == 1
    assert pipeline.failures == 0
    assert pipeline.bird.scheduler.reply_due is None
    output = next(e for e in events if e.stage == "OUTPUT")
    assert output.data["text"] == "Ответ готов обычным текстом"
    assert "reason" not in output.data
    assert any(e.stage == "FORMAT_FALLBACK" for e in events)
    assert not any(e.stage == "RETRY" for e in events)
    assert pipeline.last_request is not None
    saved = json.loads(pipeline.last_request.read_text(encoding="utf-8"))
    assert saved["decision"]["format"] == "plain_reply"
    assert saved["decision"]["reason"] == ""
