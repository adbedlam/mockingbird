from __future__ import annotations

import json
import re
from copy import copy, deepcopy
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Protocol

from mockingbird.conversation.history import History
from mockingbird.conversation.message import Message as ConversationMessage
from mockingbird.freshness import (
    history_info,
    load_latest_history,
    load_manifest,
    pair_id,
)
from mockingbird.llm.client import LLM
from mockingbird.llm.message import Message, Role
from mockingbird.persona.diary.diary import Diary
from mockingbird.persona.diary.retrieval import (
    MemorySearchConfig,
    MemorySelection,
    select_memories,
)
from mockingbird.persona.diary.runtime import DiaryMemory
from mockingbird.persona.inner.inner import InnerPrompt
from mockingbird.persona.life import PersonaLife
from mockingbird.prompts.imitation import IMITATION
from mockingbird.prompts.supress_llm_behavior import SUPRESS_LLM_BEHAVIOR
from mockingbird.scheduler.profile import BehaviorProfile, SchedulerConfig
from mockingbird.scheduler.scheduler import Action, Decision, MessageKind, Scheduler


class CompletionModel(Protocol):
    def complete(self, messages: list[Message]) -> str: ...


@dataclass(frozen=True)
class Draft:
    send: bool
    text: str
    reason: str
    format: str = "json"
    life_update: dict | None = None


class Mockingbird:
    def __init__(
        self,
        history: History,
        inner_prompt: InnerPrompt,
        diary: Diary,
        scheduler: Scheduler,
        model: CompletionModel,
        recent_messages: int = 100,
        diary_last_n: int = 30,
        *,
        memory: DiaryMemory | None = None,
        life: PersonaLife | None = None,
        archive_max_age_days: int = 365,
        memory_max_age_days: int = 365,
        memory_search: MemorySearchConfig | None = None,
    ):
        if recent_messages < 1 or diary_last_n < 1:
            raise ValueError("Context limits must be positive")
        self.history = history
        self.inner_prompt = inner_prompt
        self.diary = diary
        self.scheduler = scheduler
        self.model = model
        self.recent_messages = recent_messages
        self.diary_last_n = diary_last_n
        self.live = History(history.user, history.other, [])
        if min(archive_max_age_days, memory_max_age_days) < 1:
            raise ValueError("Context age limits must be positive")
        self.memory = memory
        self.memory_search = memory_search or (
            memory.search_config if memory is not None else MemorySearchConfig()
        )
        self.last_memory_selection: dict = {}
        self.life = life
        self.archive_max_age_days, self.memory_max_age_days = (
            archive_max_age_days,
            memory_max_age_days,
        )
        self.source_origin = ""
        self.source_info = history_info(history)
        self.persona_sources: dict = {}
        if memory is not None:
            self.diary = memory.combined()

    @classmethod
    def from_config(
        cls,
        config: dict,
        seed: int | None = None,
        model: CompletionModel | None = None,
        *,
        memory_enabled: bool = True,
    ) -> Mockingbird:
        paths = config["paths"]
        processed = Path(paths["processed"])
        history, origin = load_latest_history(config)
        if (history.user, history.other) != (config["user"], config["other"]):
            raise ValueError("Built history does not match configured participants")
        profile = BehaviorProfile(
            history, SchedulerConfig.from_dict(config.get("scheduler", {}))
        )
        settings = config["inference_model"]
        if model is None:
            model = LLM(
                settings["host"],
                settings["model"],
                settings.get("api_key"),
                settings.get("temperature", 0.7),
                timeout=settings.get("timeout_seconds", 60),
                json_mode=settings.get("json_mode", True),
            )
        limits = config.get("mockingbird", {})
        base_diary = Diary.load(processed / paths["diary_filename"])
        diary_settings = config.get("diary", {})
        context_settings = config.get("runtime_context", {})
        search_config = MemorySearchConfig.from_dict(config.get("memory_search"))
        memory = (
            DiaryMemory(
                Path(paths.get("memory", processed.parent / "memory"))
                / pair_id(history.user, history.other),
                history.user,
                history.other,
                base_diary,
                timezone=profile.config.timezone,
                update_every=diary_settings.get("update_in", 200),
                batch_size=diary_settings.get("batch_size", 200),
                context_entries=limits.get("diary_last_n", 30),
                search_config=search_config,
                recent_days=context_settings.get("memory_max_age_days", 365),
            )
            if memory_enabled and diary_settings.get("enabled", True)
            else None
        )
        bird = cls(
            history,
            InnerPrompt.load(processed / paths["inner_prompt_filename"]),
            base_diary,
            Scheduler(profile, seed),
            model,
            limits.get("history_last_n", 100),
            limits.get("diary_last_n", 30),
            memory=memory,
            life=(
                PersonaLife(
                    Path(paths.get("memory", processed.parent / "memory"))
                    / pair_id(history.user, history.other)
                    / "life",
                    history.user,
                    history.other,
                    profile.config.timezone,
                    diary_hour=config.get("life", {}).get("diary_hour", 23),
                )
                if memory_enabled and config.get("life", {}).get("enabled", False)
                else None
            ),
            archive_max_age_days=context_settings.get("archive_max_age_days", 365),
            memory_max_age_days=context_settings.get("memory_max_age_days", 365),
            memory_search=search_config,
        )
        bird.source_origin = origin
        bird.source_info = history_info(history, profile.config.timezone)
        bird.persona_sources = load_manifest(processed)
        return bird

    def receive(
        self,
        text: str,
        now: datetime,
        *,
        message_id: str | int | None = None,
        timestamp: datetime | None = None,
    ) -> None:
        if not text.strip():
            return
        self.scheduler.receive(now)
        self.live.append(
            ConversationMessage(
                self.history.other,
                timestamp or now,
                text=text.strip(),
                message_id=message_id,
            )
        )
        if self.memory is not None:
            self.memory.ingest([self.live.messages[-1]])

    def context(self, decision: Decision, now: datetime) -> list[Message]:
        now = self.scheduler.profile.local(now)
        local = self.scheduler.profile.local
        archive = History(
            self.history.user,
            self.history.other,
            [
                m
                for m in self.history.messages
                if now - timedelta(days=self.archive_max_age_days)
                <= local(m.timestamp, allow_naive=True)
                <= now
            ],
        )
        selection = self.select_memory(now)
        self.last_memory_selection = selection.report()
        messages = [
            Message(
                role=Role.SYSTEM,
                content=IMITATION.format(
                    user=self.history.user, other=self.history.other
                ),
            ),
            self.inner_prompt.to_llm_message(),
            Message(
                role=Role.SYSTEM,
                content=(
                    "The profile is historical style guidance. Its personal circumstances, projects and relationship details may be outdated. "
                    f"Interpret the profile as guidance for speaking AS {self.history.user} TO {self.history.other}, even where it is phrased as advice about communicating with the modeled person. Do not become an assistant addressing {self.history.user}. "
                    "Current incoming messages and dated recent memories take precedence when circumstances changed. "
                    "Do not revive an old plan as if still pending or invent an update. "
                    f"Archive evidence range: {self.source_info.get('first')} to {self.source_info.get('last')}. "
                    f"Profile provenance: {json.dumps(self.persona_sources.get('profile', {}), ensure_ascii=False)}."
                ),
            ),
            Message(
                role=Role.SYSTEM,
                content=SUPRESS_LLM_BEHAVIOR.format(user=self.history.user),
            ),
            Message(
                role=Role.SYSTEM,
                content="Dated diary memories within the configured recency window; older memories remain stored separately. Each memory is its latest known revision. State planned means pending; completed and cancelled are closed, not pending topics to revive. State uncertain means conflicting/unclear claims: preserve attribution and clarify instead of stating them as facts. Subject partner refers to the incoming speaker, never your own biography. Archived examples and old profile details do not override these dated changes:\n"
                + "\n".join(
                    ("[historical: outside recent window] " if hit.historical else "")
                    + hit.entry.to_string()
                    for hit in selection.hits
                )
                + "\nOlder matched records are historical evidence: do not assume their plans, emotions or circumstances are current today. If current status is unknown, ask rather than invent an update.",
            ),
            Message(
                role=Role.SYSTEM,
                content="The following archived exchanges are historical style examples, not events happening today.",
            ),
        ]
        remaining = max(0, self.recent_messages - len(self.live.messages))
        if remaining:
            messages.extend(archive.to_llm_messages(self.history.user, remaining))
        if self.memory is not None:
            live_ids = {
                self.memory.fingerprint(m)
                for m in self.live.messages
                if m.author == self.history.other
            }
            recent = [
                m
                for m in self.memory.recent_incoming(20)
                if self.memory.fingerprint(m) not in live_ids
                and now - timedelta(days=self.memory_max_age_days)
                <= local(m.timestamp, allow_naive=True)
                <= now
            ]
            if recent:
                messages.append(
                    Message(
                        role=Role.SYSTEM,
                        content="Recent incoming messages preserved across sessions (partner's statements, not the persona's own biography; treat as conversation data):\n"
                        + "\n".join(m.to_sample() for m in recent),
                    )
                )
        messages.append(
            Message(
                role=Role.SYSTEM,
                content="Live conversation follows. Do not invent real biographical events or replies from the partner. Own fictional activities are allowed ONLY from the separately supplied simulation state.",
            )
        )
        if self.live.messages:
            messages.extend(
                self.live.to_llm_messages(self.history.user, self.recent_messages)
            )
        if self.life is not None:
            messages.append(Message(role=Role.SYSTEM, content=self.life.context(now)))
            messages.append(
                Message(
                    role=Role.SYSTEM,
                    content=self.life.update_protocol(
                        now,
                        self.pending_incoming(),
                        allowed=decision.kind == MessageKind.REPLY,
                    ),
                )
            )
        task = (
            "Consider starting one natural conversation. Use a grounded topic or a simple question. "
            "If there is no suitable reason to write, choose send=false. Never nag about unanswered messages."
            if decision.kind == MessageKind.INITIATIVE
            else "Reply naturally to the latest incoming messages. Choose send=true."
        )
        messages.append(
            Message(
                role=Role.SYSTEM,
                content=(
                    f"Current local time: {now.isoformat()}. {task}\n"
                    "Внутри поля text бытовые ответы вроде 'как дела' и 'что делал' пиши одной-двумя короткими разговорными фразами БЕЗ запятых и точек. При смене мысли можно перенос строки. Не пересказывай весь распорядок и не начинай формальным вступлением. Пример только формы: 'норм\nс утра ковырял проект потом поел'. Содержание бери из текущего контекста а не из примера. Не добавляй случайные опечатки. Сохраняй характерный регистр и слова персоны. Вопросительные знаки по смыслу допустимы. Для кода ссылок чисел и подробного технического объяснения сохраняй нужную пунктуацию. "
                    "Historical emotional diary passages do not establish today's mood; do not carry an old emotional state into a casual how-are-you reply without current evidence. "
                    "Use the persona's natural voice INSIDE the text field. The outer JSON envelope is an application protocol, not part of the conversation. "
                    'Return only JSON: {"send": true, "text": "one message", "reason": "brief visible decision note", "life_update": null}. '
                    "For an agreed future plan change, replace life_update=null with the structured change described above. "
                    'For a skipped initiative use {"send": false, "text": "", "reason": "brief note", "life_update": null}. '
                    "The note describes the decision, not private reasoning."
                ),
            )
        )
        return messages

    def select_memory(self, now: datetime, query: str | None = None) -> MemorySelection:
        # Search reads current revisions, never fictional activities or generated
        # chat replies. Explicit operator queries do not become chat evidence.
        now = self.scheduler.profile.local(now)
        current = self.memory.combined() if self.memory is not None else self.diary
        explicit = query is not None
        if query is None:
            query = "\n".join(row["text"] for row in self.pending_incoming()[-3:])
        recent_incoming = [
            m.text
            for m in self.live.messages
            if m.author == self.history.other
            and m.text
            and now - timedelta(days=2)
            <= self.scheduler.profile.local(m.timestamp, allow_naive=True)
            <= now
        ]
        return select_memories(
            current.entries,
            query,
            now,
            timezone=self.scheduler.config.timezone,
            limit=self.diary_last_n,
            recent_days=self.memory_max_age_days,
            config=self.memory_search,
            user=self.history.user,
            other=self.history.other,
            context_query="" if explicit else "\n".join(recent_incoming[-3:]),
        )

    def pending_incoming(self) -> list[dict]:
        """Only the unanswered incoming burst can justify a plan change."""
        pending = []
        for message in reversed(self.live.messages):
            if message.author == self.history.user:
                break
            if message.author == self.history.other and message.text:
                pending.append(
                    {
                        "id": DiaryMemory.fingerprint(message),
                        "author": message.author,
                        "timestamp": message.timestamp.isoformat(),
                        "text": message.text,
                    }
                )
            if len(pending) >= self.recent_messages:
                break
        return list(reversed(pending))

    def compose(self, decision: Decision, now: datetime) -> Draft:
        return self.parse_draft(
            self.model.complete(self.context(decision, now)), decision.kind
        )

    @staticmethod
    def parse_draft(raw: str, kind: MessageKind | None) -> Draft:
        raw = raw.strip().lstrip("\ufeff").strip()
        if not raw:
            raise ValueError("Model returned an empty response")
        candidate = raw
        fence = re.fullmatch(
            r"```(?:json)?\s*(.*?)\s*```", raw, flags=re.DOTALL | re.IGNORECASE
        )
        if fence and (
            raw.lower().startswith("```json")
            or fence[1].lstrip().startswith(("{", "["))
        ):
            candidate = fence[1].strip()
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            # Replies already have a SEND decision from the scheduler. A plain
            # natural-language response is usable, but never invent a reason
            # or infer an initiative/diary decision from arbitrary prose.
            structured = (
                candidate.startswith(("{", "[", '"'))
                or raw.lower().startswith("```json")
                or re.search(r'"(?:send|text|reason)"\s*:', raw)
            )
            if kind == MessageKind.REPLY and not structured:
                return Draft(True, raw, "", format="plain_reply")
            raise
        if not isinstance(data, dict) or type(data.get("send")) is not bool:
            raise ValueError("Expected a structured send decision")
        text, reason = data.get("text"), data.get("reason")
        if (
            not isinstance(text, str)
            or not isinstance(reason, str)
            or not reason.strip()
        ):
            raise ValueError("Expected text and a visible decision note")
        if data["send"] and not text.strip():
            raise ValueError("Cannot send an empty message")
        if not data["send"] and kind != MessageKind.INITIATIVE:
            raise ValueError("A reply cannot be silently declined")
        update = data.get("life_update")
        if update is not None and (
            not isinstance(update, dict)
            or not data["send"]
            or kind != MessageKind.REPLY
        ):
            raise ValueError("Only a sent reply can include a structured life update")
        return Draft(data["send"], text.strip(), reason.strip(), life_update=update)

    def commit(self, decision: Decision, draft: Draft, now: datetime) -> dict | None:
        change = None
        if draft.life_update is not None and (
            self.life is None or not draft.send or decision.kind != MessageKind.REPLY
        ):
            raise ValueError("Life update is not allowed in this reply")
        if draft.send:
            if not draft.text.strip():
                raise ValueError("Cannot commit an empty message")
            # mark_sent validates the send window before touching world state.
            # If life persistence fails, preserve the pending reply for a retry.
            previous_scheduler = copy(self.scheduler.__dict__)
            previous_scheduler["planned"] = list(self.scheduler.planned)
            self.scheduler.mark_sent(decision, now)
            if draft.life_update is not None:
                assert self.life is not None
                try:
                    change = self.life.update(
                        draft.life_update, now, self.pending_incoming(), draft.text
                    )
                except Exception:
                    self.scheduler.__dict__.clear()
                    self.scheduler.__dict__.update(previous_scheduler)
                    raise
            self.live.append(
                ConversationMessage(self.history.user, now, text=draft.text)
            )
        else:
            self.scheduler.decline(decision)
        return change

    def snapshot(self) -> dict:
        return {
            "version": 1,
            "history": self.live.to_dict(),
            "scheduler": self.scheduler.to_dict(
                (self.history.user, self.history.other)
            ),
        }

    def load_snapshot(self, state: dict) -> None:
        if state.get("version") != 1:
            raise ValueError("Unsupported runtime state version")
        pair = (self.history.user, self.history.other)
        live = History.from_dict(state["history"])
        if (live.user, live.other) != pair:
            raise ValueError("Live history belongs to a different participant pair")
        scheduler = Scheduler.from_dict(
            state["scheduler"], self.scheduler.profile, pair
        )
        self.live, self.scheduler = live, scheduler
        if self.memory is not None:
            self.memory.ingest(live.messages)

    def prepare_delivery(self, decision: Decision, draft: Draft, now: datetime) -> dict:
        """Prepare an immutable, validated post-send state before touching Telegram."""
        if not draft.send or not draft.text.strip():
            raise ValueError("Cannot deliver an empty or declined draft")
        state = self.snapshot()
        scheduler = Scheduler.from_dict(
            state["scheduler"],
            self.scheduler.profile,
            (self.history.user, self.history.other),
        )
        scheduler.mark_sent(decision, now)
        live = History.from_dict(state["history"])
        live.append(ConversationMessage(self.history.user, now, text=draft.text))
        candidate = change = None
        if draft.life_update is not None:
            if self.life is None or decision.kind != MessageKind.REPLY:
                raise ValueError("Life update is not allowed in this reply")
            candidate, change = self.life.prepare_update(
                draft.life_update, now, self.pending_incoming(), draft.text
            )
        return {
            "before": state,
            "after": {
                "version": 1,
                "history": live.to_dict(),
                "scheduler": scheduler.to_dict((self.history.user, self.history.other)),
            },
            "life_day": candidate,
            "life_before": (
                deepcopy(self.life.days.get(candidate["date"]))
                if candidate and self.life
                else None
            ),
            "life_change": change,
        }

    def apply_delivery(self, prepared: dict, message_id: int) -> None:
        """Idempotently recover a confirmed send; never calls a provider."""
        state = deepcopy(prepared["after"])
        state["history"]["messages"][-1]["message_id"] = message_id
        if self.snapshot() not in (prepared["before"], state):
            raise ValueError(
                "Runtime changed since delivery; reconciliation is required"
            )
        candidate = prepared.get("life_day")
        if candidate:
            if self.life is None:
                raise ValueError("Cannot restore a life change without life state")
            key = candidate["date"]
            previous = self.life.days.get(key)
            if previous not in (prepared["life_before"], candidate):
                raise ValueError(
                    "Life plan changed since delivery; reconciliation is required"
                )
            self.life.days[key] = candidate
            try:
                self.life.save()
            except OSError:
                if previous is not None:
                    self.life.days[key] = previous
                else:
                    self.life.days.pop(key, None)
                raise
        self.load_snapshot(state)

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        temporary = directory / "runtime_state.json.tmp"
        temporary.write_text(
            json.dumps(self.snapshot(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temporary.replace(directory / "runtime_state.json")
        self.live.save(directory / "chat_history.json")
        self.scheduler.save(
            directory / "scheduler.json", (self.history.user, self.history.other)
        )

    def restore(self, directory: Path) -> None:
        snapshot = directory / "runtime_state.json"
        if snapshot.exists():
            self.load_snapshot(json.loads(snapshot.read_text(encoding="utf-8")))
            return
        state = directory / "scheduler.json"
        history = directory / "chat_history.json"
        if state.exists() != history.exists():
            raise ValueError("Runtime state is incomplete; use a new state directory")
        if state.exists():
            self.scheduler = Scheduler.load(
                state, self.scheduler.profile, (self.history.user, self.history.other)
            )
            live = History.load(history)
            if (live.user, live.other) != (self.history.user, self.history.other):
                raise ValueError("Live history belongs to a different participant pair")
            self.live = live
            if self.memory is not None:
                self.memory.ingest(live.messages)


def main() -> None:
    from mockingbird.terminal import main as terminal_main

    terminal_main()


if __name__ == "__main__":
    main()
