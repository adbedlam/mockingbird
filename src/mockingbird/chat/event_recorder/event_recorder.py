from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

from mockingbird.chat.event_recorder.chat_transcript import ChatTranscript
from mockingbird.chat.event_recorder.event_journal import EventJournal


@dataclass(frozen=True)
class PipelineEvent:
    at: str
    stage: str
    message: str
    data: dict = field(default_factory=dict)


class EventRecorder:
    def __init__(
        self,
        directory: Path,
        run_id: str,
        on_event: Callable[[PipelineEvent], None],
        clock: Callable[[], datetime],
    ):
        self.directory = directory
        self.journal = EventJournal(directory / "decisions.jsonl")
        self.transcript = ChatTranscript(directory / "chat_transcript.md")
        self.run_id = run_id
        self.on_event = on_event
        self.clock = clock
        self.event_number = 0

    def emit(self, stage: str, message: str, **data: object) -> None:
        self.event_number += 1
        data = {**data, "run_id": self.run_id, "event_number": self.event_number}
        event = PipelineEvent(self.clock().isoformat(), stage, message, data)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.journal.append(
            asdict(event),
            durable=stage in {"OUTPUT", "ERROR", "DELIVERY_UNKNOWN", "TG_ACK", "STOP"},
        )
        self.transcript.append(event.at, stage, message, data)
        self.on_event(event)
