from __future__ import annotations

import json
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from mockingbird.conversation.history import History
from mockingbird.conversation.message import Message as ConversationMessage
from mockingbird.mockingbird.draft import Draft
from mockingbird.persona.diary.runtime import DiaryMemory
from mockingbird.persona.life import PersonaLife
from mockingbird.scheduler.scheduler import Decision, MessageKind, Scheduler


@dataclass(frozen=True)
class RuntimeAccessors:
    history: Callable[[], History]
    live: Callable[[], History]
    set_live: Callable[[History], None]
    scheduler: Callable[[], Scheduler]
    set_scheduler: Callable[[Scheduler], None]
    memory: Callable[[], DiaryMemory | None]


class RuntimeStore:
    def __init__(self, accessors: RuntimeAccessors):
        self._access = accessors

    def snapshot(self) -> dict:
        history = self._access.history()
        return {
            "version": 1,
            "history": self._access.live().to_dict(),
            "scheduler": self._access.scheduler().to_dict(
                (history.user, history.other)
            ),
        }

    def load_snapshot(self, state: dict) -> None:
        if state.get("version") != 1:
            raise ValueError("Unsupported runtime state version")
        history = self._access.history()
        pair = (history.user, history.other)
        live = History.from_dict(state["history"])
        if (live.user, live.other) != pair:
            raise ValueError("Live history belongs to a different participant pair")
        scheduler = Scheduler.from_dict(
            state["scheduler"], self._access.scheduler().profile, pair
        )
        self._access.set_live(live)
        self._access.set_scheduler(scheduler)
        memory = self._access.memory()
        if memory is not None:
            memory.ingest(live.messages)

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        temporary = directory / "runtime_state.json.tmp"
        temporary.write_text(
            json.dumps(self.snapshot(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temporary.replace(directory / "runtime_state.json")
        self._access.live().save(directory / "chat_history.json")
        history = self._access.history()
        self._access.scheduler().save(
            directory / "scheduler.json", (history.user, history.other)
        )

    def restore(self, directory: Path) -> None:
        snapshot = directory / "runtime_state.json"
        if snapshot.exists():
            self.load_snapshot(json.loads(snapshot.read_text(encoding="utf-8")))
            return
        state = directory / "scheduler.json"
        history_path = directory / "chat_history.json"
        if state.exists() != history_path.exists():
            raise ValueError("Runtime state is incomplete; use a new state directory")
        if state.exists():
            history = self._access.history()
            pair = (history.user, history.other)
            scheduler = Scheduler.load(
                state, self._access.scheduler().profile, pair
            )
            live = History.load(history_path)
            if (live.user, live.other) != pair:
                raise ValueError(
                    "Live history belongs to a different participant pair"
                )
            self._access.set_scheduler(scheduler)
            self._access.set_live(live)
            memory = self._access.memory()
            if memory is not None:
                memory.ingest(live.messages)


@dataclass(frozen=True)
class DeliveryAccessors:
    runtime: RuntimeStore
    history: Callable[[], History]
    scheduler: Callable[[], Scheduler]
    life: Callable[[], PersonaLife | None]
    pending_incoming: Callable[[], list[dict]]


class DeliveryCoordinator:
    def __init__(self, accessors: DeliveryAccessors):
        self._access = accessors

    def prepare_delivery(
        self, decision: Decision, draft: Draft, now: datetime
    ) -> dict:
        """Validate a post-send state before an external delivery attempt."""
        access = self._access
        if not draft.send or not draft.text.strip():
            raise ValueError("Cannot deliver an empty or declined draft")
        state = access.runtime.snapshot()
        history = access.history()
        pair = (history.user, history.other)
        scheduler = Scheduler.from_dict(
            state["scheduler"], access.scheduler().profile, pair
        )
        scheduler.mark_sent(decision, now)
        live = History.from_dict(state["history"])
        live.append(ConversationMessage(history.user, now, text=draft.text))
        candidate = change = None
        life_before = None
        if draft.life_update is not None:
            life = access.life()
            if life is None or decision.kind != MessageKind.REPLY:
                raise ValueError("Life update is not allowed in this reply")
            candidate, change = life.prepare_update(
                draft.life_update, now, access.pending_incoming(), draft.text
            )
            life_before = deepcopy(life.days.get(candidate["date"]))
        return {
            "before": state,
            "after": {
                "version": 1,
                "history": live.to_dict(),
                "scheduler": scheduler.to_dict(pair),
            },
            "life_day": candidate,
            "life_before": life_before,
            "life_change": change,
        }

    def apply_delivery(self, prepared: dict, message_id: int) -> None:
        """Idempotently apply a confirmed delivery without contacting a provider."""
        access = self._access
        state = deepcopy(prepared["after"])
        state["history"]["messages"][-1]["message_id"] = message_id
        if access.runtime.snapshot() not in (prepared["before"], state):
            raise ValueError(
                "Runtime changed since delivery; reconciliation is required"
            )
        candidate = prepared.get("life_day")
        if candidate:
            life = access.life()
            if life is None:
                raise ValueError("Cannot restore a life change without life state")
            key = candidate["date"]
            previous = life.days.get(key)
            if previous not in (prepared["life_before"], candidate):
                raise ValueError(
                    "Life plan changed since delivery; reconciliation is required"
                )
            life.days[key] = candidate
            try:
                life.save()
            except OSError:
                if previous is not None:
                    life.days[key] = previous
                else:
                    life.days.pop(key, None)
                raise
        access.runtime.load_snapshot(state)
