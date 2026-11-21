from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from uuid import uuid4


class RequestStore:
    def __init__(self, directory: Path):
        self.directory = directory

    def latest(self) -> Path | None:
        paths = list((self.directory / "requests").glob("*.json"))
        return max(paths, key=lambda path: path.stat().st_mtime) if paths else None

    def create(
        self,
        run_id: str,
        at: datetime,
        kind: str,
        model: dict,
        messages: list,
        memory_selection: dict | None,
    ) -> Path:
        request_id = uuid4().hex
        path = self.directory / "requests" / f"{request_id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "id": request_id,
                    "run_id": run_id,
                    "at": at.isoformat(),
                    "kind": kind,
                    "model": model,
                    "messages": messages,
                    "status": "pending",
                    "memory_selection": memory_selection,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return path

    def update(self, path: Path, **changes: object) -> None:
        data = json.loads(path.read_text(encoding="utf-8"))
        data.update(changes)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temporary.replace(path)

    def abandon(self, path: Path, at: datetime) -> None:
        self.update(path, status="abandoned", closed_at=at.isoformat())
