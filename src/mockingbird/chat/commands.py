from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Protocol

from mockingbird.chat.tasks import PendingRequest
from mockingbird.mockingbird.mockingbird import Mockingbird
from mockingbird.observability import export_bundle


class CommandContext(Protocol):
    bird: Mockingbird
    directory: Path
    closed: bool
    inflight: PendingRequest | None
    max_generations: int | None
    generations: int
    paused: bool
    failures: int
    diary_requested: bool
    last_request: Path | None

    def clock(self) -> datetime: ...

    def emit(self, stage: str, message: str, **data: object) -> None: ...

    def status(self) -> str: ...

    def advance_life(self, *, checkpoint: bool = False) -> None: ...


class CommandRouter:
    def __init__(self, context: CommandContext):
        self.context = context

    def handle(self, text: str) -> bool:
        """Return False only for /quit; commands never become model input."""
        context = self.context
        command = text.strip()
        if command == "/quit":
            return False
        if command in ("/help", "/?"):
            context.emit(
                "HELP",
                "/status — состояние; /plan — инициативы; /life — смоделированные занятия; /memory [тема] — локальный поиск памяти; /context — файл точного запроса; /context full — весь запрос; /history — последние реплики; /diary — дневник и смоделированные дни; /diary update — обновить по новым входящим; /pause — пауза; /resume — продолжить; /quit — выход.",
            )
        elif command == "/export":
            memory_directory = (
                context.bird.memory.directory
                if context.bird.memory
                else (context.bird.life.directory.parent if context.bird.life else None)
            )
            path = export_bundle(context.directory, memory_directory)
            context.emit("EXPORT", f"Архив отладки с приватной перепиской: {path}")
        elif command == "/status":
            context.emit("STATUS", context.status())
        elif command == "/life":
            life = context.bird.life
            context.emit(
                "LIFE_STATE",
                (
                    life.view(context.clock())
                    if life
                    else "Симуляция дня отключена в настройках."
                ),
            )
        elif command == "/memory" or command.startswith("/memory "):
            query = command[len("/memory") :].strip()
            selected = context.bird.memory_query.select(context.clock(), query or None)
            context.emit(
                "MEMORY_SEARCH",
                f"Локальный поиск: выбрано {len(selected.hits)} из {selected.considered} актуальных записей; {selected.chars}/{selected.max_chars} символов. Запрос: {selected.query or 'текущий разговор'}",
                text="\n\n".join(
                    f"{hit.entry.to_string()}\nВыбрано: {hit.reason}; совпавшие основы: {', '.join(hit.matched_terms) or 'нет'}; историческое: {'да' if hit.historical else 'нет'}"
                    for hit in selected.hits
                )
                or "Совпадений и подходящего недавнего контекста нет",
                selection=selected.report(),
            )
        elif command == "/plan":
            scheduler = context.bird.scheduler
            context.emit(
                "PLAN",
                f"Дата: {scheduler.day}. Оставшиеся инициативы: "
                + (", ".join(t.isoformat() for t in scheduler.planned) or "нет")
                + ". Ответ: "
                + (
                    scheduler.reply_due.isoformat()
                    if scheduler.reply_due
                    else "не ожидается"
                ),
            )
        elif command in ("/context", "/context full"):
            if context.last_request is None:
                context.emit(
                    "CONTEXT",
                    "Запросов в этой папке пока нет: контекст будет сохранён перед первой генерацией.",
                )
            else:
                value = (
                    context.last_request.read_text(encoding="utf-8")
                    if command.endswith(" full")
                    else str(context.last_request)
                )
                context.emit("CONTEXT", value)
        elif command in ("/diary", "/diary update"):
            if command == "/diary" and context.bird.life is not None:
                context.advance_life(checkpoint=True)
                context.emit("DAILY_DIARY", context.bird.life.diary_view())
            memory = context.bird.memory
            if memory is None:
                context.emit(
                    "DIARY",
                    "Накопление памяти в этом запуске отключено. Исходный дневник:\n"
                    + str(context.bird.diary),
                )
            elif command == "/diary":
                memory.render()
                context.emit(
                    "DIARY",
                    f"В очереди {len(memory.pending)} новых входящих. Дневник: {memory.directory / 'diary.md'}.\n"
                    + str(memory.combined()),
                )
            elif memory.pending:
                context.diary_requested = True
                context.emit(
                    "DIARY_QUEUED",
                    "Запрошено обновление дневника. Сначала завершим ответ собеседнику; затем отдельно покажем добавленные записи или причину пропуска.",
                )
            else:
                context.emit(
                    "DIARY_SKIP",
                    "Новых входящих для дневника нет; повторно обработанные сообщения не отправляем модели.",
                )
        elif command == "/history":
            rows = [
                f"{message.timestamp.isoformat()} | {message.author}: {message.text or ''}"
                for message in context.bird.live.messages[-10:]
            ]
            context.emit(
                "HISTORY",
                "\n".join(rows) if rows else "В текущем диалоге пока нет сообщений.",
            )
        elif command == "/pause":
            context.paused = True
            if context.inflight is not None:
                context.inflight.cancelled = True
            context.emit(
                "PAUSED",
                "Новые запросы остановлены. Результат уже выполняющегося запроса также не будет выведен. /resume — продолжить.",
            )
        elif command == "/resume":
            if (
                context.max_generations is not None
                and context.generations >= context.max_generations
            ):
                context.emit(
                    "LIMIT",
                    "Лимит запросов этого запуска исчерпан. Для нового лимита перезапусти процесс с --max-generations.",
                )
            else:
                context.paused = False
                context.failures = 0
                context.emit(
                    "RESUMED",
                    "Генерация возобновлена. Просроченные инициативы не догоняем.",
                )
        else:
            context.emit(
                "HELP",
                "Неизвестная команда. /help — список команд. Чтобы передать текст со слешем модели, начни строку с //.",
            )
        return True
