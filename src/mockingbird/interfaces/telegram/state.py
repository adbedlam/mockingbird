"""Durable state shared by the official Telegram Business interface."""

from __future__ import annotations

import json
import secrets
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mockingbird.mockingbird.mockingbird import Mockingbird


class SetupError(RuntimeError):
    """An error occured during setup."""


def participant_id(value: object) -> int:
    text = str(value).removeprefix("user")
    if not text.isdecimal() or int(text) < 1:
        raise SetupError("Для Telegram нужны числовые user_id и other_id из экспорта.")
    return int(text)


class DeliveryJournal:
    def __init__(self, directory: Path, identity: tuple[int, int]):
        directory.mkdir(parents=True, exist_ok=True)
        self.directory = directory
        self.db = sqlite3.connect(directory / "transport.sqlite3")
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.executescript(
            "CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);"
            "CREATE TABLE IF NOT EXISTS outbox ("
            "id TEXT PRIMARY KEY, status TEXT NOT NULL, text TEXT NOT NULL,"
            "payload TEXT NOT NULL, random_id INTEGER UNIQUE NOT NULL,"
            "message_id INTEGER, error TEXT, created_at TEXT NOT NULL);"
        )
        stored = self.get("identity")
        if stored is not None and stored != list(identity):
            self.close()
            raise SetupError(
                "Папка Telegram принадлежит другому аккаунту или собеседнику."
            )
        self.set("identity", list(identity))

    def get(self, key: str, default: Any = None) -> Any:
        row = self.db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else default

    def _set(self, key: str, value: object) -> None:
        self.db.execute(
            "INSERT OR REPLACE INTO meta VALUES (?, ?)",
            (key, json.dumps(value, ensure_ascii=False)),
        )

    def set(self, key: str, value: object) -> None:
        with self.db:
            self._set(key, value)

    def save_state(self, mockingbird: Mockingbird, paused: bool) -> None:
        with self.db:
            self._set("state", mockingbird.runtime_store.snapshot())
            self._set("paused", paused)

    def prepare(self, identity: str, text: str, payload: dict) -> int:
        random_id = secrets.randbits(63) or 1
        with self.db:
            self.db.execute(
                "INSERT INTO outbox VALUES (?, 'prepared', ?, ?, ?, NULL, NULL, ?)",
                (
                    identity,
                    text,
                    json.dumps(payload, ensure_ascii=False),
                    random_id,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
        return random_id

    def update(
        self,
        identity: str,
        status: str,
        *,
        message_id: int | None = None,
        error: str = "",
    ) -> None:
        with self.db:
            self.db.execute(
                "UPDATE outbox SET status=?, message_id=COALESCE(?, message_id), error=? WHERE id=?",
                (status, message_id, error, identity),
            )

    def unresolved(self) -> list[dict]:
        return [
            dict(row)
            for row in self.db.execute(
                "SELECT * FROM outbox WHERE status IN ('prepared','sending','unknown','sent') ORDER BY created_at"
            )
        ]

    def complete(self, identity: str, mockingbird: Mockingbird) -> None:
        # Commit the acknowledged post-send state and outbox marker together.
        with self.db:
            self._set("state", mockingbird.runtime_store.snapshot())
            self.db.execute(
                "UPDATE outbox SET status='applied' WHERE id=?", (identity,)
            )

    def known_message(self, message_id: int) -> bool:
        return (
            self.db.execute(
                "SELECT 1 FROM outbox WHERE message_id=? AND status='applied'",
                (message_id,),
            ).fetchone()
            is not None
        )

    def report(self) -> None:
        path = self.directory / "delivery_report.json"
        data = {
            "identity": self.get("identity"),
            "cursor": self.get("cursor"),
            "paused": self.get("paused", False),
            "outbox": [
                dict(row)
                for row in self.db.execute("SELECT * FROM outbox ORDER BY created_at")
            ],
        }
        temporary = path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temporary.replace(path)

    def close(self) -> None:
        self.db.close()
