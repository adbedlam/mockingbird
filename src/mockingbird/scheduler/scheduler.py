from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from enum import StrEnum
from pathlib import Path
from random import Random

from mockingbird.scheduler.profile import (
    BehaviorProfile,
    validate_scheduler_settings,
)


class MessageKind(StrEnum):
    INITIATIVE = "initiative"
    REPLY = "reply"


class Action(StrEnum):
    WAIT = "wait"
    SEND = "send"
    SKIP = "skip"


@dataclass(frozen=True)
class Decision:
    action: Action
    at: datetime
    reason: str
    kind: MessageKind | None = None
    due_at: datetime | None = None
    revision: int = 0


class Scheduler:
    """A clock-independent state machine. SEND is committed only after success.

    Call tick(now), compose a message, then mark_sent(decision, now). Neither
    tick nor construction sends messages or invokes an LLM.
    """

    def __init__(self, profile: BehaviorProfile, seed: int | None = None):
        self.profile = profile
        self.config = profile.config
        self.rng = Random(seed)
        self.day: date | None = None
        self.planned: list[datetime] = []
        self.initiations_sent = 0
        self.reply_due: datetime | None = None
        self.last_incoming: datetime | None = None
        self.last_outgoing: datetime | None = None
        self.awaiting_reply = False
        self.busy_until: datetime | None = None
        self.revision = 0

    def _ensure_day(self, now: datetime) -> None:
        if self.day != now.date():
            self.revision += 1
            self.day = now.date()
            self.planned = self.profile.plan(self.day, self.rng)
            self.initiations_sent = 0

    def receive(self, now: datetime) -> None:
        now = self.profile.local(now)
        if self.last_incoming is not None and now < self.last_incoming:
            raise ValueError(
                "Incoming messages must be observed in chronological order"
            )
        self.last_incoming = now
        self.revision += 1
        self.awaiting_reply = False
        # Multiple incoming fragments coalesce into one response; allow at least
        # min_reply_delay after the most recent fragment before generating it.
        if self.reply_due is None:
            delay = (
                timedelta(seconds=self.config["min_reply_delay_seconds"])
                if self.config["reply_mode"] == "immediate"
                else self.profile.reply_delay(self.rng)
            )
            self.reply_due = now + delay
        self.reply_due = max(
            self.reply_due,
            now + timedelta(seconds=self.config["min_reply_delay_seconds"]),
        )
        if self.config["reply_mode"] == "history":
            self.reply_due = self.profile.next_active(self.reply_due)

    def set_busy(self, until: datetime | None) -> None:
        self.busy_until = self.profile.local(until) if until is not None else None
        self.revision += 1

    def tick(self, now: datetime) -> Decision:
        now = self.profile.local(now)
        self._ensure_day(now)
        if self.reply_due is not None and self.reply_due <= now:
            if self.busy_until is not None and now < self.busy_until:
                self.reply_due = (
                    self.profile.next_active(self.busy_until)
                    if self.config["reply_mode"] == "history"
                    else self.busy_until
                )
            elif self.config["reply_mode"] == "history" and not self.profile.is_active(
                now
            ):
                self.reply_due = self.profile.next_active(now)
            else:
                return Decision(
                    Action.SEND,
                    now,
                    "Настало время ответить на входящие сообщения",
                    MessageKind.REPLY,
                    self.reply_due,
                    self.revision,
                )
        if self.planned and self.planned[0] <= now:
            due = self.planned[0]
            reason = None
            if (now - due).total_seconds() > self.config["max_lateness_seconds"]:
                reason = (
                    "Просроченная инициатива пропущена, догонять расписание не нужно"
                )
            elif self.awaiting_reply:
                reason = "Предыдущая инициатива ещё без ответа"
            elif self.reply_due is not None:
                reason = "Сначала нужно ответить на входящие сообщения"
            elif self.initiations_sent >= self.config["max_initiations_per_day"]:
                reason = "Достигнут дневной лимит инициатив"
            elif self.busy_until is not None and now < self.busy_until:
                reason = "Персона занята"
            elif not self.profile.is_active(now):
                reason = "Сейчас нехарактерное для переписки время"
            elif (
                self.last_outgoing is not None
                and (now - self.last_outgoing).total_seconds()
                < self.config["min_initiation_gap_seconds"]
            ):
                reason = "С последнего сообщения персоны прошло слишком мало времени"
            elif (
                self.last_incoming is not None
                and (now - self.last_incoming).total_seconds()
                < self.config["silence_before_initiation_seconds"]
            ):
                reason = "Недавно уже общались, новая инициатива не нужна"
            if reason is not None:
                self.planned.pop(0)
                return Decision(Action.SKIP, now, reason, MessageKind.INITIATIVE, due)
            return Decision(
                Action.SEND,
                now,
                "Подходящее время по истории; лимит и паузы соблюдены",
                MessageKind.INITIATIVE,
                due,
                self.revision,
            )
        candidates = self.planned[:1] + (
            [self.reply_due] if self.reply_due is not None else []
        )
        return Decision(
            Action.WAIT,
            now,
            "Ожидаем следующего события",
            due_at=min(candidates) if candidates else None,
        )

    def _check_pending(self, decision: Decision) -> None:
        expected = (
            self.reply_due
            if decision.kind == MessageKind.REPLY
            else (self.planned[0] if self.planned else None)
        )
        if (
            decision.action != Action.SEND
            or expected is None
            or decision.due_at != expected
            or decision.revision != self.revision
        ):
            raise ValueError("Decision is no longer pending")

    def is_pending(self, decision: Decision) -> bool:
        try:
            self._check_pending(decision)
        except ValueError:
            return False
        return True

    def mark_sent(self, decision: Decision, now: datetime) -> None:
        self._check_pending(decision)
        now = self.profile.local(now)
        immediate_reply = (
            decision.kind == MessageKind.REPLY
            and self.config["reply_mode"] == "immediate"
        )
        if not immediate_reply and (
            not self.profile.is_active(now) or now.date() != decision.at.date()
        ):
            raise ValueError("The send window expired while composing the message")
        if decision.kind == MessageKind.INITIATIVE:
            self.planned.pop(0)
            self.initiations_sent += 1
            self.awaiting_reply = True
        else:
            self.reply_due = None
        self.last_outgoing = now
        self.revision += 1

    def decline(self, decision: Decision) -> None:
        self._check_pending(decision)
        if decision.kind != MessageKind.INITIATIVE:
            raise ValueError("Only an initiative may be declined")
        self.planned.pop(0)
        self.revision += 1

    def retry_later(self, decision: Decision, now: datetime) -> None:
        self._check_pending(decision)
        due = self.profile.local(now) + timedelta(minutes=5)
        if decision.kind != MessageKind.REPLY or self.config["reply_mode"] == "history":
            due = self.profile.next_active(due)
        if decision.kind == MessageKind.REPLY:
            self.reply_due = due
        else:
            # Retry this attempt; the original daily budget is unchanged.
            self.planned[0] = due
            self.planned.sort()
        self.revision += 1

    def to_dict(self, pair: tuple[str, str]) -> dict:
        return {
            "version": 1,
            "pair": list(pair),
            "config": self.config.copy(),
            "day": self.day.isoformat() if self.day else None,
            "planned": [t.isoformat() for t in self.planned],
            "initiations_sent": self.initiations_sent,
            "awaiting_reply": self.awaiting_reply,
            "revision": self.revision,
            **{
                name: getattr(self, name).isoformat() if getattr(self, name) else None
                for name in (
                    "reply_due",
                    "last_incoming",
                    "last_outgoing",
                    "busy_until",
                )
            },
        }

    def save(self, path: Path, pair: tuple[str, str]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(self.to_dict(pair), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary.replace(path)

    @classmethod
    def load(
        cls, path: Path, profile: BehaviorProfile, pair: tuple[str, str]
    ) -> Scheduler:
        return cls.from_dict(
            json.loads(path.read_text(encoding="utf-8")), profile, pair
        )

    @classmethod
    def from_dict(
        cls, data: dict, profile: BehaviorProfile, pair: tuple[str, str]
    ) -> Scheduler:
        if data.get("version") != 1 or data.get("pair") != list(pair):
            raise ValueError(
                "Scheduler state version or participant pair does not match"
            )
        saved_config = data.get("config", {})
        if not isinstance(saved_config, dict):
            raise ValueError("Invalid scheduler configuration in state")
        saved_config = dict(saved_config)
        # Older snapshots predate reply_mode; their behavior was "history".
        saved_config.setdefault("reply_mode", "history")
        validate_scheduler_settings(saved_config)
        if saved_config != profile.config:
            raise ValueError("Scheduler configuration changed; use a new state file")
        scheduler = cls(profile)
        scheduler.day = date.fromisoformat(data["day"]) if data["day"] else None
        scheduler.planned = [
            profile.local(datetime.fromisoformat(t)) for t in data["planned"]
        ]
        scheduler.initiations_sent = data["initiations_sent"]
        scheduler.awaiting_reply = data["awaiting_reply"]
        scheduler.revision = data["revision"]
        for name in ("reply_due", "last_incoming", "last_outgoing", "busy_until"):
            value = data[name]
            setattr(
                scheduler,
                name,
                profile.local(datetime.fromisoformat(value)) if value else None,
            )
        return scheduler
