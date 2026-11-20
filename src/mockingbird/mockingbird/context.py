from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta

from mockingbird.conversation.history import History
from mockingbird.conversation.message import Message as ConversationMessage
from mockingbird.llm.message import Message, Role
from mockingbird.persona.diary.retrieval import MemorySelection
from mockingbird.persona.diary.runtime import DiaryMemory
from mockingbird.persona.inner.inner import InnerPrompt
from mockingbird.persona.life import PersonaLife
from mockingbird.prompts.chat import (
    ARCHIVE_CONTEXT,
    LIVE_CONTEXT,
    LIVE_CONVERSATION_CONTEXT,
    MEMORY_CONTEXT,
    PROFILE_CONTEXT,
    response_instructions,
)
from mockingbird.prompts.imitation import IMITATION
from mockingbird.prompts.supress_llm_behavior import SUPRESS_LLM_BEHAVIOR
from mockingbird.scheduler.scheduler import Decision, MessageKind


@dataclass(frozen=True)
class ContextAccessors:
    history: Callable[[], History]
    live: Callable[[], History]
    inner_prompt: Callable[[], InnerPrompt]
    life: Callable[[], PersonaLife | None]
    select_memory: Callable[[datetime], MemorySelection]
    pending_incoming: Callable[[], list[dict]]
    recent_incoming: Callable[[int], list[ConversationMessage]]
    local_time: Callable[[datetime], datetime]
    recent_messages: Callable[[], int]
    archive_max_age_days: Callable[[], int]
    memory_max_age_days: Callable[[], int]
    source_info: Callable[[], dict]
    persona_sources: Callable[[], dict]


class ContextBuilder:
    def __init__(self, accessors: ContextAccessors):
        self.accessors = accessors

    def build(self, decision: Decision, now: datetime) -> list[Message]:
        access = self.accessors
        history = access.history()
        live = access.live()
        now = access.local_time(now)
        local = access.local_time
        archive = History(
            history.user,
            history.other,
            [
                message
                for message in history.messages
                if now - timedelta(days=access.archive_max_age_days())
                <= local(message.timestamp)
                <= now
            ],
        )
        selection = access.select_memory(now)
        messages = [
            Message(
                role=Role.SYSTEM,
                content=IMITATION.format(user=history.user, other=history.other),
            ),
            access.inner_prompt().to_llm_message(),
            Message(
                role=Role.SYSTEM,
                content=PROFILE_CONTEXT.format(
                    user=history.user,
                    other=history.other,
                    first=access.source_info().get("first"),
                    last=access.source_info().get("last"),
                    sources=json.dumps(
                        access.persona_sources().get("profile", {}),
                        ensure_ascii=False,
                    ),
                ),
            ),
            Message(
                role=Role.SYSTEM,
                content=SUPRESS_LLM_BEHAVIOR.format(
                    user=history.user,
                    other=history.other,
                ),
            ),
            Message(
                role=Role.SYSTEM,
                content=MEMORY_CONTEXT.format(
                    entries="\n".join(
                        (
                            "[historical: outside recent window] "
                            if hit.historical
                            else ""
                        )
                        + hit.entry.to_string()
                        for hit in selection.hits
                    )
                ),
            ),
            Message(role=Role.SYSTEM, content=ARCHIVE_CONTEXT),
        ]
        remaining = max(0, access.recent_messages() - len(live.messages))
        if remaining:
            messages.extend(archive.to_llm_messages(history.user, remaining))
        live_ids = {
            DiaryMemory.fingerprint(message)
            for message in live.messages
            if message.author == history.other
        }
        recent = [
            message
            for message in access.recent_incoming(20)
            if message.author == history.other
            and DiaryMemory.fingerprint(message) not in live_ids
            and now - timedelta(days=access.memory_max_age_days())
            <= local(message.timestamp)
            <= now
        ]
        if recent:
            messages.append(
                Message(
                    role=Role.SYSTEM,
                    content=LIVE_CONTEXT.format(
                        messages="\n".join(message.to_sample() for message in recent)
                    ),
                )
            )
        messages.append(Message(role=Role.SYSTEM, content=LIVE_CONVERSATION_CONTEXT))
        if live.messages:
            messages.extend(
                live.to_llm_messages(history.user, access.recent_messages())
            )
        life = access.life()
        if life is not None:
            messages.append(Message(role=Role.SYSTEM, content=life.context(now)))
            messages.append(
                Message(
                    role=Role.SYSTEM,
                    content=life.update_protocol(
                        now,
                        access.pending_incoming(),
                        allowed=decision.kind == MessageKind.REPLY,
                    ),
                )
            )
        messages.append(
            Message(role=Role.SYSTEM, content=response_instructions(now, decision.kind))
        )
        return messages
