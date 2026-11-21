from __future__ import annotations

import json
import os
from pathlib import Path


class EventJournal:
    def __init__(self, path: Path):
        self.path = path

    def append(self, event: dict, *, durable: bool) -> None:
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, ensure_ascii=False) + "\n")
            stream.flush()
            if durable:
                os.fsync(stream.fileno())
