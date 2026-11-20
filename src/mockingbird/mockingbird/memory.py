from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timedelta

from mockingbird.conversation.history import History
from mockingbird.persona.diary.diary import Diary
from mockingbird.persona.diary.retrieval import (
    MemorySearchConfig,
    MemorySelection,
    select_memories,
)
from mockingbird.persona.diary.runtime import DiaryMemory


class MemoryQuery:
    def __init__(
        self,
        live: Callable[[], History],
        memory: Callable[[], DiaryMemory | None],
        diary: Callable[[], Diary],
        local_time: Callable[[datetime], datetime],
        timezone: str,
        *,
        limit: int,
        recent_days: int,
        recent_messages: int,
        config: MemorySearchConfig,
    ):
        self._live = live
        self._memory = memory
        self._diary = diary
        self._local_time = local_time
        self._timezone = timezone
        self._limit = limit
        self._recent_days = recent_days
        self._recent_messages = recent_messages
        self._config = config

    def pending_incoming(self) -> list[dict]:
        """Return the unanswered incoming burst that can justify a plan change."""
        live = self._live()
        pending = []
        for message in reversed(live.messages):
            if message.author == live.user:
                break
            if message.author == live.other and message.text:
                pending.append(
                    {
                        "id": DiaryMemory.fingerprint(message),
                        "author": message.author,
                        "timestamp": message.timestamp.isoformat(),
                        "text": message.text,
                    }
                )
            if len(pending) >= self._recent_messages:
                break
        return list(reversed(pending))

    def select(self, now: datetime, query: str | None = None) -> MemorySelection:
        now = self._local_time(now)
        live = self._live()
        memory = self._memory()
        current = memory.combined() if memory is not None else self._diary()
        explicit = query is not None
        if query is None:
            query = "\n".join(row["text"] for row in self.pending_incoming()[-3:])
        recent_incoming = [
            message.text
            for message in live.messages
            if message.author == live.other
            and message.text
            and now - timedelta(days=2) <= self._local_time(message.timestamp) <= now
        ]
        return select_memories(
            current.entries,
            query,
            now,
            timezone=self._timezone,
            limit=self._limit,
            recent_days=self._recent_days,
            config=self._config,
            user=live.user,
            other=live.other,
            context_query="" if explicit else "\n".join(recent_incoming[-3:]),
        )
