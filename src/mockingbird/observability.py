"""Local run IDs, technical logs and explicit private diagnostic exports."""

from __future__ import annotations

import json
import logging
import os
import platform
import re
import sys
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from uuid import uuid4
from zipfile import ZIP_DEFLATED, ZipFile


def redact(text: str) -> str:
    for key, value in os.environ.items():
        if (
            value
            and len(value) >= 6
            and (key.endswith(("API_KEY", "API_HASH", "PASSWORD", "TOKEN")))
        ):
            text = text.replace(value, "[REDACTED]")
    return text


class PrivateFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        return redact(super().format(record))


def start_run(directory: Path, model_info: dict) -> tuple[str, logging.Logger]:
    directory.mkdir(parents=True, exist_ok=True)
    run_id = uuid4().hex
    logger = logging.getLogger(f"mockingbird.run.{run_id}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    handler = RotatingFileHandler(
        directory / "technical.log", maxBytes=5_000_000, backupCount=3, encoding="utf-8"
    )
    handler.setFormatter(PrivateFormatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    manifest = {
        "run_id": run_id,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.split()[0],
        "platform": platform.system(),
        "model": {
            k: v
            for k, v in model_info.items()
            if k in ("host", "model", "temperature", "json_mode")
        },
    }
    runs = directory / "runs"
    runs.mkdir(exist_ok=True)
    (runs / f"{run_id}.json").write_text(
        redact(json.dumps(manifest, ensure_ascii=False, indent=2)), encoding="utf-8"
    )
    return run_id, logger


def export_bundle(directory: Path, memory: Path | None = None) -> Path:
    """An allowlist keeps auth sessions, environment files and media out of ZIPs.

    Conversation text and raw model payloads are intentionally included. This
    is a private debugging bundle, not a public error report.
    """
    if not directory.is_dir():
        raise ValueError("Runtime directory does not exist")
    output = directory / "exports"
    output.mkdir(exist_ok=True)
    target = output / f"debug-{datetime.now():%Y%m%d-%H%M%S}-{uuid4().hex[:8]}.zip"
    allowed = {
        "decisions.jsonl",
        "chat_transcript.md",
        "runtime_state.json",
        "chat_history.json",
        "scheduler.json",
        "delivery_report.json",
        "business_status.json",
        "diary.json",
        "diary.md",
        "memory.json",
        "life.json",
        "daily_diary.md",
        "plan_changes.md",
        "diary_updates.md",
        "diary_history.md",
    }
    sources: list[tuple[Path, str]] = [(directory, "runtime")]
    if memory is not None and memory.is_dir():
        sources.append((memory, "memory"))
    with ZipFile(target, "w", ZIP_DEFLATED) as archive:
        archive.writestr(
            "README.txt",
            "Private chat, exact model requests and diary are included. No Telegram session or .env files.\n",
        )
        for root, prefix in sources:
            for path in sorted(root.rglob("*")):
                if not path.is_file() or path.is_symlink():
                    continue
                relative = path.relative_to(root)
                if any(
                    part.startswith(".") or part == "exports" for part in relative.parts
                ):
                    continue
                accepted = (
                    path.name in allowed
                    or re.fullmatch(r"technical\.log(?:\.[123])?", path.name)
                    or (
                        relative.parts[0] in ("requests", "runs")
                        and path.suffix == ".json"
                    )
                )
                if accepted:
                    archive.writestr(
                        f"{prefix}/{relative.as_posix()}",
                        redact(path.read_text(encoding="utf-8")),
                    )
    return target
