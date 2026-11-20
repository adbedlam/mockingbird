from __future__ import annotations

from copy import copy
from datetime import datetime
from pathlib import Path
from typing import Protocol

from mockingbird.conversation.history import History
from mockingbird.conversation.identity import pair_id
from mockingbird.conversation.message import Message as ConversationMessage
from mockingbird.llm.client import LLM
from mockingbird.llm.message import Message
from mockingbird.mockingbird.context import ContextAccessors, ContextBuilder
from mockingbird.mockingbird.draft import Draft, parse_draft
from mockingbird.mockingbird.memory import MemoryQuery
from mockingbird.mockingbird.runtime import (
    DeliveryAccessors,
    DeliveryCoordinator,
    RuntimeAccessors,
    RuntimeStore,
)
from mockingbird.persona.diary.diary import Diary
from mockingbird.persona.diary.retrieval import MemorySearchConfig
from mockingbird.persona.diary.runtime import DiaryMemory
from mockingbird.persona.inner.inner import InnerPrompt
from mockingbird.persona.life import LifePrompt, PersonaLife
from mockingbird.persona.source import load_manifest
from mockingbird.preprocessing.telegram.source import load_latest_history
from mockingbird.scheduler.profile import BehaviorProfile
from mockingbird.scheduler.scheduler import Decision, MessageKind, Scheduler


class CompletionModel(Protocol):
    def complete(self, messages: list[Message]) -> str: ...


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
        self.life_prompt = LifePrompt(content="")
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
        self.source_info = History.history_info(history)
        self.persona_sources: dict = {}
        if memory is not None:
            self.diary = memory.combined()
        self.memory_query = MemoryQuery(
            live=lambda: self.live,
            memory=lambda: self.memory,
            diary=lambda: self.diary,
            local_time=lambda value: self.scheduler.profile.local(
                value, allow_naive=True
            ),
            timezone=self.scheduler.config["timezone"],
            limit=self.diary_last_n,
            recent_days=self.memory_max_age_days,
            recent_messages=self.recent_messages,
            config=self.memory_search,
        )
        self.context_builder = ContextBuilder(
            ContextAccessors(
                history=lambda: self.history,
                live=lambda: self.live,
                inner_prompt=lambda: self.inner_prompt,
                life=lambda: self.life,
                select_memory=self._select_context_memory,
                pending_incoming=self.memory_query.pending_incoming,
                recent_incoming=lambda limit: (
                    self.memory.recent_incoming(limit)
                    if self.memory is not None
                    else []
                ),
                local_time=lambda value: self.scheduler.profile.local(
                    value, allow_naive=True
                ),
                recent_messages=lambda: self.recent_messages,
                archive_max_age_days=lambda: self.archive_max_age_days,
                memory_max_age_days=lambda: self.memory_max_age_days,
                source_info=lambda: self.source_info,
                persona_sources=lambda: self.persona_sources,
            )
        )
        self.runtime_store = RuntimeStore(
            RuntimeAccessors(
                history=lambda: self.history,
                live=lambda: self.live,
                set_live=lambda value: setattr(self, "live", value),
                scheduler=lambda: self.scheduler,
                set_scheduler=lambda value: setattr(self, "scheduler", value),
                memory=lambda: self.memory,
            )
        )
        self.delivery_coordinator = DeliveryCoordinator(
            DeliveryAccessors(
                runtime=self.runtime_store,
                history=lambda: self.history,
                scheduler=lambda: self.scheduler,
                life=lambda: self.life,
                pending_incoming=self.memory_query.pending_incoming,
            )
        )

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
        profile = BehaviorProfile(history, config.get("scheduler", {}))
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
        runtime = Path(paths.get("runtime", processed.parent / "runtime"))
        memory_root = Path(paths.get("memory", runtime / "memory"))
        life_prompt_path = processed / paths.get(
            "life_prompt_filename", "life_prompt.md"
        )
        life_prompt = (
            LifePrompt.load(life_prompt_path)
            if life_prompt_path.exists()
            else LifePrompt(content="")
        )
        memory = (
            DiaryMemory(
                memory_root / pair_id(history.user, history.other),
                history.user,
                history.other,
                base_diary,
                timezone=profile.config["timezone"],
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
                    memory_root / pair_id(history.user, history.other) / "life",
                    history.user,
                    history.other,
                    profile.config["timezone"],
                    diary_hour=config.get("life", {}).get("diary_hour", 23),
                    life_prompt=life_prompt.content,
                )
                if memory_enabled and config.get("life", {}).get("enabled", False)
                else None
            ),
            archive_max_age_days=context_settings.get("archive_max_age_days", 365),
            memory_max_age_days=context_settings.get("memory_max_age_days", 365),
            memory_search=search_config,
        )
        bird.life_prompt = life_prompt
        bird.source_origin = origin
        bird.source_info = History.history_info(history, profile.config["timezone"])
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

    def _select_context_memory(self, now: datetime):
        selection = self.memory_query.select(now)
        self.last_memory_selection = selection.report()
        return selection

    def compose(self, decision: Decision, now: datetime) -> Draft:
        return parse_draft(
            self.model.complete(self.context_builder.build(decision, now)),
            decision.kind,
        )

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
                        draft.life_update,
                        now,
                        self.memory_query.pending_incoming(),
                        draft.text,
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


def main() -> None:
    from mockingbird.interfaces.terminal import main as terminal_main

    terminal_main()


if __name__ == "__main__":
    main()
