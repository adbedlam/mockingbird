from __future__ import annotations

import json
import os
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

from mockingbird.chat.presenter import reason_label


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
        self.run_id = run_id
        self.on_event = on_event
        self.clock = clock
        self.event_number = 0

    def emit(self, stage: str, message: str, **data: object) -> None:
        self.event_number += 1
        data = {**data, "run_id": self.run_id, "event_number": self.event_number}
        event = PipelineEvent(self.clock().isoformat(), stage, message, data)
        self.directory.mkdir(parents=True, exist_ok=True)
        with (self.directory / "decisions.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(asdict(event), ensure_ascii=False) + "\n")
            stream.flush()
            if stage in {"OUTPUT", "ERROR", "DELIVERY_UNKNOWN", "TG_ACK", "STOP"}:
                os.fsync(stream.fileno())
        with (self.directory / "chat_transcript.md").open(
            "a", encoding="utf-8"
        ) as stream:
            stream.write(f"\n## {event.at} — {stage}\n\n{message}\n")
            if "text" in data:
                stream.write(
                    "\n"
                    + "\n".join("> " + line for line in str(data["text"]).splitlines())
                    + "\n"
                )
            if "reason" in data:
                stream.write(
                    "\n" + reason_label(data) + ": " + str(data["reason"]) + "\n"
                )
            if "request_path" in data:
                stream.write(
                    "\nКонтекст и сырой ответ: " + str(data["request_path"]) + "\n"
                )
        self.on_event(event)
