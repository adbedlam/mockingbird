from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Callable

from mockingbird.mockingbird.mockingbird import Mockingbird


@dataclass(frozen=True)
class Snapshot:
    day: date | None
    planned: tuple[datetime, ...]
    reply_due: datetime | None
    revision: int
    initiations_sent: int
    busy_until: datetime | None
    live_message_count: int

    @classmethod
    def capture(cls, mockingbird: Mockingbird) -> Snapshot:
        scheduler = mockingbird.scheduler
        return cls(
            scheduler.day,
            tuple(scheduler.planned),
            scheduler.reply_due,
            scheduler.revision,
            scheduler.initiations_sent,
            scheduler.busy_until,
            len(mockingbird.live.messages),
        )


class SnapshotStore:
    def __init__(self) -> None:
        self._saved: Snapshot | None = None

    def persist(
        self,
        mockingbird: Mockingbird,
        directory: Path,
        state_sink: Callable[[], None] | None,
    ) -> None:
        snapshot = Snapshot.capture(mockingbird)
        if snapshot == self._saved:
            return
        if state_sink is not None:
            state_sink()
        mockingbird.runtime_store.save(directory)
        self._saved = snapshot
