from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import BinaryIO, Self

from mockingbird.chat.event_recorder.event_recorder import PipelineEvent
from mockingbird.chat.presenter import reason_label

if sys.platform == "win32":
    import msvcrt
else:
    import fcntl


class SessionLock:
    """OS lock released automatically on process exit, including a crash."""

    def __init__(self, directory: Path):
        self.path = directory / ".session.lock"
        self.stream: BinaryIO | None = None

    def __enter__(self) -> Self:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        stream = self.path.open("a+b")
        if self.path.stat().st_size == 0:
            stream.write(b" ")
            stream.flush()
        stream.seek(0)
        try:
            if sys.platform == "win32":
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            stream.close()
            raise RuntimeError(
                "Эта папка состояния уже открыта другим процессом. Выбери другой --state-dir."
            ) from exc
        self.stream = stream
        return self

    def __exit__(self, *args: object) -> None:
        if self.stream is not None:
            if sys.platform == "win32":
                self.stream.seek(0)
                msvcrt.locking(self.stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(self.stream.fileno(), fcntl.LOCK_UN)
            self.stream.close()
            self.stream = None


def render(event: PipelineEvent) -> None:
    stamp = datetime.fromisoformat(event.at).strftime("%H:%M:%S")
    print(f"[{stamp}] {event.stage} | {event.message}", flush=True)
    if "text" in event.data:
        print(str(event.data["text"]), flush=True)
    if "reason" in event.data:
        print(
            "  " + reason_label(event.data) + ": " + str(event.data["reason"]),
            flush=True,
        )
    if event.stage == "REQUEST" and event.data.get("request_path"):
        print("  Контекст: " + str(event.data["request_path"]), flush=True)
