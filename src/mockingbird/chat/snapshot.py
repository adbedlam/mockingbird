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
    def capture(cls, bird: Mockingbird) -> Snapshot:
        scheduler = bird.scheduler
        return cls(
            scheduler.day,
            tuple(scheduler.planned),
            scheduler.reply_due,
            scheduler.revision,
            scheduler.initiations_sent,
            scheduler.busy_until,
            len(bird.live.messages),
        )


class SnapshotStore:
    def __init__(self) -> None:
        self._saved: Snapshot | None = None

    def persist(
        self,
        bird: Mockingbird,
        directory: Path,
        state_sink: Callable[[], None] | None,
    ) -> None:
        snapshot = Snapshot.capture(bird)
        if snapshot == self._saved:
            return
        if state_sink is not None:
            state_sink()
        bird.runtime_store.save(directory)
        self._saved = snapshot
