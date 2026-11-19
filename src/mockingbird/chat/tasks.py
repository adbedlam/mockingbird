from __future__ import annotations

from concurrent.futures import Future
from dataclasses import dataclass
from pathlib import Path

from mockingbird.persona.diary.runtime import DiaryBatch
from mockingbird.scheduler.scheduler import Decision


@dataclass(frozen=True)
class ReplyTask:
    decision: Decision


@dataclass(frozen=True)
class DiaryTask:
    batch: DiaryBatch


@dataclass(frozen=True)
class LifePlanTask:
    date: str


RequestTask = ReplyTask | DiaryTask | LifePlanTask


@dataclass
class ModelResult:
    raw: str = ""
    error: str = ""


@dataclass
class PendingRequest:
    task: RequestTask
    path: Path
    started: float
    heartbeat: float
    future: Future[ModelResult]
    cancelled: bool = False
