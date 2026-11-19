import json
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest
from config_helpers import scheduler_settings

from mockingbird.conversation.history import History
from mockingbird.conversation.message import Message
from mockingbird.mockingbird.mockingbird import Draft, Mockingbird
from mockingbird.persona.diary.diary import Diary
from mockingbird.persona.inner.inner import InnerPrompt
from mockingbird.scheduler.observations import conversation_bursts
from mockingbird.scheduler.profile import (
    BehaviorProfile,
    validate_scheduler_settings,
)
from mockingbird.scheduler.scheduler import Action, MessageKind, Scheduler

ZONE = ZoneInfo("Europe/Moscow")


def at(day: int = 17, hour: int = 9, minute: int = 0, second: int = 0) -> datetime:
    return datetime(2026, 11, day, hour, minute, second, tzinfo=ZONE)


@pytest.fixture
def history() -> History:
    messages = []
    for offset in range(14):
        day = datetime(2026, 10, 1, tzinfo=ZONE) + timedelta(days=offset)
        for author, hour, minute, second in (
            ("Partner", 8, 0, 0),
            ("Persona", 9, 0, 0),
            ("Persona", 9, 0, 20),
            ("Partner", 9, 1, 0),
            ("Persona", 9, 1, 10),
            ("Persona", 17, 0, 0),
            ("Partner", 17, 2, 0),
            ("Persona", 17, 2, 20),
        ):
            messages.append(
                Message(
                    author,
                    day.replace(hour=hour, minute=minute, second=second),
                    text="example",
                )
            )
    return History("Persona", "Partner", messages)


@pytest.fixture
def profile(history: History) -> BehaviorProfile:
    return BehaviorProfile(history, scheduler_settings())


def scheduled(profile: BehaviorProfile, when: datetime) -> Scheduler:
    scheduler = Scheduler(profile, seed=7)
    scheduler.day = when.date()
    scheduler.planned = [when]
    return scheduler


def test_plan_uses_observed_hours_and_daily_limit(profile: BehaviorProfile):
    scheduler = Scheduler(profile, seed=42)
    for offset in range(30):
        day = date(2026, 11, 1) + timedelta(days=offset)
        plan = profile.plan(day, scheduler.rng)
        assert len(plan) == 2
        assert all(profile.is_active(t) and t.hour in (9, 17) for t in plan)
        assert (plan[1] - plan[0]).total_seconds() >= 14400


def test_seed_reproduces_plan(profile: BehaviorProfile):
    first, second = Scheduler(profile, 12), Scheduler(profile, 12)
    first.tick(at(hour=0))
    second.tick(at(hour=0))
    assert first.planned == second.planned


def test_no_observed_initiatives_means_no_forced_greeting():
    h = History(
        "Persona",
        "Partner",
        [
            Message("Partner", at(), text="hi"),
            Message("Persona", at(second=20), text="hello"),
        ],
    )
    scheduler = Scheduler(BehaviorProfile(h, scheduler_settings()), 7)
    scheduler.tick(at())
    assert scheduler.planned == []


def test_unanswered_initiative_blocks_later_days(
    profile: BehaviorProfile, tmp_path: Path
):
    scheduler = scheduled(profile, at())
    decision = scheduler.tick(at())
    scheduler.mark_sent(decision, at())
    scheduler.save(tmp_path / "state.json", ("Persona", "Partner"))
    restored = Scheduler.load(tmp_path / "state.json", profile, ("Persona", "Partner"))
    restored.tick(at(day=18, hour=0))
    decision = restored.tick(restored.planned[0])
    assert decision.action == Action.SKIP
    assert "без ответа" in decision.reason


def test_tick_does_not_commit_and_duplicate_commit_is_rejected(
    profile: BehaviorProfile,
):
    scheduler = scheduled(profile, at())
    decision = scheduler.tick(at())
    assert scheduler.initiations_sent == 0
    assert scheduler.tick(at()) == decision
    scheduler.mark_sent(decision, at())
    assert scheduler.initiations_sent == 1
    with pytest.raises(ValueError, match="no longer pending"):
        scheduler.mark_sent(decision, at())


def test_incoming_invalidates_prepared_initiative(profile: BehaviorProfile):
    scheduler = scheduled(profile, at())
    decision = scheduler.tick(at())
    scheduler.receive(at(second=1))
    with pytest.raises(ValueError, match="no longer pending"):
        scheduler.mark_sent(decision, at(second=2))
    assert scheduler.initiations_sent == 0


def test_multiple_incoming_fragments_get_one_reply(profile: BehaviorProfile):
    scheduler = Scheduler(profile, 7)
    scheduler.receive(at())
    scheduler.receive(at(second=3))
    assert scheduler.reply_due is not None
    assert scheduler.reply_due >= at(second=8)
    decision = scheduler.tick(scheduler.reply_due)
    assert decision.kind == MessageKind.REPLY
    scheduler.mark_sent(decision, decision.at)
    assert scheduler.reply_due is None
    assert not scheduler.awaiting_reply


def test_nighttime_incoming_waits_for_observed_activity(profile: BehaviorProfile):
    scheduler = Scheduler(profile, 7)
    scheduler.receive(at(hour=23))
    assert scheduler.reply_due == at(day=18, hour=9)
    assert scheduler.tick(at(hour=23, minute=5)).action != Action.SEND
    assert scheduler.tick(at(day=18, hour=9)).kind == MessageKind.REPLY


def test_night_owl_is_not_forced_into_daytime():
    h = History(
        "Persona",
        "Partner",
        [Message("Partner", at(hour=0)), Message("Persona", at(hour=1))],
    )
    p = BehaviorProfile(h, scheduler_settings())
    assert p.is_active(at(hour=1))
    assert not p.is_active(at(hour=12))


def test_busy_defers_reply_and_cancels_initiative(profile: BehaviorProfile):
    scheduler = scheduled(profile, at())
    scheduler.set_busy(at(hour=17))
    assert scheduler.tick(at()).action == Action.SKIP
    scheduler.receive(at())
    decision = scheduler.tick(at(minute=1))
    assert decision.action == Action.WAIT
    assert scheduler.reply_due == at(hour=17)


def test_daily_cap_and_recent_interaction(profile: BehaviorProfile):
    scheduler = scheduled(profile, at())
    scheduler.initiations_sent = 2
    assert "лимит" in scheduler.tick(at()).reason
    scheduler = scheduled(profile, at())
    scheduler.last_outgoing = at(hour=8)
    assert "мало времени" in scheduler.tick(at()).reason


def test_missed_slots_are_not_sent_on_restart(profile: BehaviorProfile):
    scheduler = scheduled(profile, at())
    assert "Просроченная" in scheduler.tick(at(hour=17)).reason
    assert scheduler.initiations_sent == 0


def test_state_round_trip_and_pair_config_checks(
    profile: BehaviorProfile, tmp_path: Path
):
    scheduler = scheduled(profile, at())
    scheduler.receive(at())
    state = tmp_path / "state.json"
    scheduler.save(state, ("Persona", "Partner"))
    restored = Scheduler.load(state, profile, ("Persona", "Partner"))
    assert restored.planned == scheduler.planned
    assert restored.reply_due == scheduler.reply_due
    with pytest.raises(ValueError, match="participant pair"):
        Scheduler.load(state, profile, ("Someone", "Else"))
    profile.config["max_initiations_per_day"] = 1
    with pytest.raises(ValueError, match="configuration changed"):
        Scheduler.load(state, profile, ("Persona", "Partner"))


def test_retry_does_not_use_send_budget(profile: BehaviorProfile):
    scheduler = scheduled(profile, at())
    decision = scheduler.tick(at())
    scheduler.retry_later(decision, at())
    assert scheduler.initiations_sent == 0
    assert scheduler.planned[0] == at(minute=5)


def test_imported_order_and_unknown_authors_do_not_corrupt_observations(
    history: History,
):
    expected = [
        burst.messages[0]
        for burst in conversation_bursts(history)
        if burst.author == history.user and burst.initiated
    ]
    mixed = History(
        history.user,
        history.other,
        list(reversed(history.messages)) + [Message("Forwarded", at())],
    )
    observed = [
        burst.messages[0]
        for burst in conversation_bursts(mixed)
        if burst.author == mixed.user and burst.initiated
    ]
    assert observed == expected
    assert len(expected) == 28
    assert all(b.author != "Forwarded" for b in conversation_bursts(mixed))


def test_runtime_requires_timezone(profile: BehaviorProfile):
    with pytest.raises(ValueError, match="timezone"):
        Scheduler(profile).tick(datetime(2026, 11, 17))


class FakeModel:
    def __init__(self, response: str):
        self.response = response
        self.calls = 0

    def complete(self, messages: list) -> str:
        self.calls += 1
        assert messages[-1]["role"] == "system"
        return self.response


@pytest.mark.parametrize("send", [True, False])
def test_model_can_send_or_decline_without_changing_source_history(
    history: History, profile: BehaviorProfile, send: bool
):
    scheduler = scheduled(profile, at())
    model = FakeModel(
        json.dumps(
            {
                "send": send,
                "text": "Привет" if send else "",
                "reason": "Есть тема" if send else "Нет темы",
            }
        )
    )
    bird = Mockingbird(history, InnerPrompt("Style"), Diary(), scheduler, model)
    decision = scheduler.tick(at())
    draft = bird.compose(decision, at())
    bird.commit(decision, draft, at())
    assert scheduler.initiations_sent == int(send)
    assert len(bird.live.messages) == int(send)
    assert len(history.messages) == 112
    assert model.calls == 1


def test_bad_model_output_keeps_pending_decision(
    history: History, profile: BehaviorProfile
):
    scheduler = scheduled(profile, at())
    bird = Mockingbird(
        history, InnerPrompt("Style"), Diary(), scheduler, FakeModel('{"send": true,')
    )
    decision = scheduler.tick(at())
    with pytest.raises(ValueError):
        bird.compose(decision, at())
    assert scheduler.initiations_sent == 0
    assert scheduler.planned == [at()]


def test_runtime_restart_keeps_incoming_and_reply(
    history: History, profile: BehaviorProfile, tmp_path: Path
):
    bird = Mockingbird(
        history, InnerPrompt("Style"), Diary(), Scheduler(profile, 7), FakeModel("")
    )
    bird.receive("Привет", at())
    bird.save(tmp_path)
    restored = Mockingbird(
        history, InnerPrompt("Style"), Diary(), Scheduler(profile), FakeModel("")
    )
    restored.restore(tmp_path)
    assert restored.live.messages == bird.live.messages
    assert restored.scheduler.reply_due == bird.scheduler.reply_due


def test_empty_message_cannot_be_committed(history: History, profile: BehaviorProfile):
    scheduler = scheduled(profile, at())
    bird = Mockingbird(history, InnerPrompt("Style"), Diary(), scheduler, FakeModel(""))
    decision = scheduler.tick(at())
    with pytest.raises(ValueError, match="empty message"):
        bird.commit(decision, Draft(True, "", "invalid"), at())
    assert scheduler.initiations_sent == 0


def test_unknown_scheduler_settings_are_not_silently_ignored():
    with pytest.raises(ValueError, match="Unknown scheduler settings"):
        validate_scheduler_settings(scheduler_settings(max_initiations_per_dya=0))


def test_missing_scheduler_settings_are_not_defaulted():
    with pytest.raises(ValueError, match="Missing scheduler settings"):
        validate_scheduler_settings({})
