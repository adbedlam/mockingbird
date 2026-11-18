"""Observable local chat orchestration; terminal and model are replaceable.

Only the main thread touches history, scheduler state, and audit files. The
daemon worker receives an immutable request snapshot and returns raw output.
"""

from __future__ import annotations

import json
import os
import queue
import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from pathlib import Path
from uuid import uuid4

from mockingbird.mockingbird.mockingbird import Draft, Mockingbird
from mockingbird.observability import export_bundle, start_run
from mockingbird.persona.diary.runtime import DiaryBatch
from mockingbird.scheduler.scheduler import Action, Decision


@dataclass(frozen=True)
class PipelineEvent:
    at: str
    stage: str
    message: str
    data: dict = field(default_factory=dict)

    @property
    def reason_label(self) -> str:
        return {
            "application": "Проверка приложения",
            "batch": "Пояснение модели к пакету",
        }.get(str(self.data.get("reason_source", "")), "Решение модели")


@dataclass
class PendingRequest:
    decision: Decision | None
    path: Path
    started: float
    heartbeat: float
    cancelled: bool = False
    diary_batch: DiaryBatch | None = None
    life_date: str | None = None


@dataclass(frozen=True)
class ModelResult:
    raw: str = ""
    error: str = ""


class DeliveryBlocked(RuntimeError):
    """Delivery needs operator attention; never schedule a fresh automatic retry."""


class DeliverySuperseded(RuntimeError):
    """New Telegram messages made the draft stale immediately before sending."""


class DeliveryDeferred(RuntimeError):
    """A read failed before sending. Wait for transport recovery, not operator input."""


class ChatPipeline:
    def __init__(
        self,
        bird: Mockingbird,
        directory: Path,
        on_event: Callable[[PipelineEvent], None],
        *,
        model_info: dict | None = None,
        clock: Callable[[], datetime] | None = None,
        max_failures: int = 3,
        max_generations: int | None = None,
        delivery: (
            Callable[[Decision, Draft, datetime, Path], dict | None] | None
        ) = None,
        state_sink: Callable[[], None] | None = None,
    ):
        if max_failures < 1 or (max_generations is not None and max_generations < 1):
            raise ValueError("Generation limits must be positive")
        self.bird = bird
        self.directory = directory
        self.delivery = delivery
        self.delivery_blocked = False
        self.delivery_deferred = False
        self.state_sink = state_sink
        self.run_id, self.logger = start_run(directory, model_info or {})
        self.event_number = 0
        self.on_event = on_event
        self.model_info = model_info or {}
        self.clock = clock or (lambda: datetime.now(bird.scheduler.profile.zone))
        self.max_failures = max_failures
        self.max_generations = max_generations
        self.generations = 0
        self.failures = 0
        self.paused = False
        self.closed = False
        self.inflight: PendingRequest | None = None
        self._results: queue.Queue[ModelResult] = queue.Queue()
        self._saved_state: tuple | None = None
        self._shown_plan: date | None = None
        self._shown_wait: tuple | None = None
        self.last_request: Path | None = None
        self.diary_requested = False
        previous = list((directory / "requests").glob("*.json"))
        if previous:
            self.last_request = max(previous, key=lambda p: p.stat().st_mtime)

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
                    "\n" + event.reason_label + ": " + str(data["reason"]) + "\n"
                )
            if "request_path" in data:
                stream.write(
                    "\nКонтекст и сырой ответ: " + str(data["request_path"]) + "\n"
                )
        self.on_event(event)

    def persist(self) -> None:
        scheduler = self.bird.scheduler
        state = (
            scheduler.day,
            tuple(scheduler.planned),
            scheduler.reply_due,
            scheduler.revision,
            scheduler.initiations_sent,
            scheduler.busy_until,
            len(self.bird.live.messages),
        )
        if state != self._saved_state:
            if self.state_sink is not None:
                self.state_sink()
            self.bird.save(self.directory)
            self._saved_state = state

    def start(self, *, run_step: bool = True) -> None:
        bird = self.bird
        self.emit(
            "SESSION",
            f"Персона: {bird.history.user}; собеседник: {bird.history.other}. "
            f"Модель: {self.model_info.get('model', type(bird.model).__name__)}. "
            f"Часовой пояс: {bird.scheduler.config.timezone}.",
        )
        mode = bird.scheduler.config.reply_mode
        self.emit(
            "MODE",
            (
                f"Быстрые ответы: через {bird.scheduler.config.min_reply_delay_seconds} сек. после последнего фрагмента, даже вне активных часов. Инициативы — по обычным правилам."
                if mode == "immediate"
                else "Ответы и инициативы следуют времени из истории. Долгое ожидание будет показано явно; для обычного тестового диалога есть --fast-replies."
            ),
        )
        self.emit(
            "RESTORED",
            f"Загружено: {len(bird.history.messages)} архивных сообщений, "
            f"{len(bird.diary.entries)} записей дневника, {len(bird.live.messages)} сообщений текущего диалога. "
            f"Журнал: {self.directory / 'chat_transcript.md'}.",
        )
        self.emit(
            "FRESHNESS",
            f"Архив: {bird.source_info.get('first')} — {bird.source_info.get('last')}. "
            f"Источник: {bird.source_origin or 'передан напрямую'}. "
            f"Профиль: {bird.persona_sources.get('profile', {}).get('last', 'дата исходных данных неизвестна; --refresh-persona пересоберёт его')}. "
            f"Способ построения: {bird.persona_sources.get('profile', {}).get('method', 'model_or_legacy')}. "
            f"Недавние записи — за {bird.memory_max_age_days} дней. "
            + (
                "Поиск по всей сохранённой памяти, старые совпадения помечены historical. "
                if bird.memory_search.enabled
                and bird.memory_search.max_age_days is None
                else (
                    f"Предел возраста поиска: {bird.memory_search.max_age_days} дней. "
                    if bird.memory_search.enabled
                    else "Используется только недавний контекст. "
                )
            )
            + f"Поиск {'включён' if bird.memory_search.enabled else 'выключен'}, бюджет выбранных воспоминаний: {bird.memory_search.max_chars} символов.",
        )
        if bird.memory is not None:
            self.emit(
                "DIARY",
                f"Актуальных записей слоя памяти: {len(bird.memory.entries)}; входящих в очереди: {len(bird.memory.pending)}/{bird.memory.update_every}. "
                f"/diary — посмотреть, /diary update — обновить вручную. Файл: {bird.memory.directory / 'diary.md'}.",
            )
        if bird.life is not None:
            self.emit(
                "LIFE",
                f"Вымышленный день роли: один план на дату. /life — занятия; /diary — дневник. "
                f"Итог в {bird.life.diary_hour}:00 по местному времени и при выходе. "
                f"Файл: {bird.life.directory / 'daily_diary.md'}.",
            )
        if run_step:
            self.step()

    def receive(
        self,
        text: str,
        received_at: datetime | None = None,
        *,
        message_id: str | int | None = None,
        timestamp: datetime | None = None,
    ) -> None:
        if self.closed or not text.strip():
            return
        at = received_at or self.clock()
        self.bird.receive(text, at, message_id=message_id, timestamp=timestamp)
        self.persist()
        self.emit(
            "INPUT",
            f"Принято сообщение от {self.bird.history.other}.",
            text=text.strip(),
            message_id=message_id,
            source_timestamp=timestamp.isoformat() if timestamp else None,
        )
        due = self.bird.scheduler.reply_due
        self.emit(
            "SCHEDULE",
            f"Ответ запланирован на {due.isoformat() if due else '—'}. "
            "Новые фрагменты до ответа объединяются в одну реплику.",
        )
        if self.inflight is not None and self.inflight.decision is not None:
            self.emit(
                "SUPERSEDED",
                "Пришло новое сообщение во время генерации. Старый результат будет отброшен; затем соберём обновлённый контекст.",
            )

    def status(self) -> str:
        s = self.bird.scheduler
        return "\n".join(
            [
                f"Время: {self.clock().isoformat()} | режим ответов: {s.config.reply_mode}",
                f"Запрос к модели: {'выполняется' if self.inflight else 'нет'} | генерации: {self.generations}/{self.max_generations if self.max_generations is not None else 'без лимита'}",
                f"Пауза: {'да' if self.paused else 'нет'} | ошибок подряд: {self.failures}/{self.max_failures}",
                f"Ожидаемый ответ: {s.reply_due.isoformat() if s.reply_due else 'нет'}",
                f"Инициатив отправлено: {s.initiations_sent}/{s.config.max_initiations_per_day} | ожидаем ответ собеседника: {'да' if s.awaiting_reply else 'нет'}",
                "Оставшиеся слоты: "
                + (", ".join(t.isoformat() for t in s.planned) or "нет"),
                "Симуляция дня: "
                + (
                    f"план {'сохранён' if self.bird.life.local(self.clock()).date().isoformat() in self.bird.life.days else 'пока отсутствует'}; дневник ежедневно в {self.bird.life.diary_hour}:00 и при выходе"
                    if self.bird.life
                    else "отключена"
                ),
                "Память: "
                + (
                    f"{len(self.bird.memory.entries)} актуальных записей слоя памяти; {len(self.bird.memory.pending)}/{self.bird.memory.update_every} входящих в очереди"
                    if self.bird.memory
                    else "только исходный дневник"
                ),
            ]
        )

    def command(self, text: str) -> bool:
        """Return False only for /quit; commands never become model input."""
        command = text.strip()
        if command == "/quit":
            return False
        if command in ("/help", "/?"):
            self.emit(
                "HELP",
                "/status — состояние; /plan — инициативы; /life — смоделированные занятия; /memory [тема] — локальный поиск памяти; /context — файл точного запроса; /context full — весь запрос; /history — последние реплики; /diary — дневник и смоделированные дни; /diary update — обновить по новым входящим; /pause — пауза; /resume — продолжить; /quit — выход.",
            )
        elif command == "/export":
            memory_directory = (
                self.bird.memory.directory
                if self.bird.memory
                else (self.bird.life.directory.parent if self.bird.life else None)
            )
            path = export_bundle(self.directory, memory_directory)
            self.emit("EXPORT", f"Архив отладки с приватной перепиской: {path}")
        elif command == "/status":
            self.emit("STATUS", self.status())
        elif command == "/life":
            life = self.bird.life
            self.emit(
                "LIFE_STATE",
                (
                    life.view(self.clock())
                    if life
                    else "Симуляция дня отключена в настройках."
                ),
            )
        elif command == "/memory" or command.startswith("/memory "):
            query = command[len("/memory") :].strip()
            selected = self.bird.select_memory(self.clock(), query or None)
            self.emit(
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
            s = self.bird.scheduler
            self.emit(
                "PLAN",
                f"Дата: {s.day}. Оставшиеся инициативы: "
                + (", ".join(t.isoformat() for t in s.planned) or "нет")
                + ". Ответ: "
                + (s.reply_due.isoformat() if s.reply_due else "не ожидается"),
            )
        elif command in ("/context", "/context full"):
            if self.last_request is None:
                self.emit(
                    "CONTEXT",
                    "Запросов в этой папке пока нет: контекст будет сохранён перед первой генерацией.",
                )
            else:
                value = (
                    self.last_request.read_text(encoding="utf-8")
                    if command.endswith(" full")
                    else str(self.last_request)
                )
                self.emit("CONTEXT", value)
        elif command in ("/diary", "/diary update"):
            if command == "/diary" and self.bird.life is not None:
                self._advance_life(checkpoint=True)
                self.emit("DAILY_DIARY", self.bird.life.diary_view())
            memory = self.bird.memory
            if memory is None:
                self.emit(
                    "DIARY",
                    "Накопление памяти в этом запуске отключено. Исходный дневник:\n"
                    + str(self.bird.diary),
                )
            elif command == "/diary":
                memory.render()
                self.emit(
                    "DIARY",
                    f"В очереди {len(memory.pending)} новых входящих. Дневник: {memory.directory / 'diary.md'}.\n"
                    + str(memory.combined()),
                )
            elif memory.pending:
                self.diary_requested = True
                self.emit(
                    "DIARY_QUEUED",
                    "Запрошено обновление дневника. Сначала завершим ответ собеседнику; затем отдельно покажем добавленные записи или причину пропуска.",
                )
            else:
                self.emit(
                    "DIARY_SKIP",
                    "Новых входящих для дневника нет; повторно обработанные сообщения не отправляем модели.",
                )
        elif command == "/history":
            rows = [
                f"{m.timestamp.isoformat()} | {m.author}: {m.text or ''}"
                for m in self.bird.live.messages[-10:]
            ]
            self.emit(
                "HISTORY",
                "\n".join(rows) if rows else "В текущем диалоге пока нет сообщений.",
            )
        elif command == "/pause":
            self.paused = True
            if self.inflight is not None:
                self.inflight.cancelled = True
            self.emit(
                "PAUSED",
                "Новые запросы остановлены. Результат уже выполняющегося запроса также не будет выведен. /resume — продолжить.",
            )
        elif command == "/resume":
            if (
                self.max_generations is not None
                and self.generations >= self.max_generations
            ):
                self.emit(
                    "LIMIT",
                    "Лимит запросов этого запуска исчерпан. Для нового лимита перезапусти процесс с --max-generations.",
                )
            else:
                self.paused = False
                self.failures = 0
                self.emit(
                    "RESUMED",
                    "Генерация возобновлена. Просроченные инициативы не догоняем.",
                )
        else:
            self.emit(
                "HELP",
                "Неизвестная команда. /help — список команд. Чтобы передать текст со слешем модели, начни строку с //.",
            )
        return True

    def _update_request(self, pending: PendingRequest, **changes: object) -> None:
        data = json.loads(pending.path.read_text(encoding="utf-8"))
        data.update(changes)
        temporary = pending.path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temporary.replace(pending.path)

    def _begin(self, decision: Decision, now: datetime) -> None:
        messages = self.bird.context(decision, now)
        self._launch(
            messages,
            now,
            str(decision.kind),
            decision=decision,
            memory_selection=self.bird.last_memory_selection,
        )

    def _begin_diary(self, now: datetime) -> None:
        memory = self.bird.memory
        assert memory is not None
        batch = memory.prepare(now)
        self.diary_requested = False
        self._launch(
            batch.messages,
            now,
            "diary",
            diary_batch=batch,
            memory_selection=batch.memory_selection,
        )

    def _begin_life(self, now: datetime) -> None:
        life = self.bird.life
        assert life is not None
        self._launch(
            life.prepare(now, self.bird.inner_prompt.content),
            now,
            "life_plan",
            life_date=life.local(now).date().isoformat(),
        )

    def _advance_life(self, *, checkpoint: bool = False) -> None:
        if self.bird.life is not None:
            for stage, message in self.bird.life.advance(
                self.clock(), checkpoint=checkpoint
            ):
                self.emit(stage, message)

    def _launch(
        self,
        messages: list,
        now: datetime,
        kind: str,
        *,
        decision: Decision | None = None,
        diary_batch: DiaryBatch | None = None,
        life_date: str | None = None,
        memory_selection: dict | None = None,
    ) -> None:
        request_id = uuid4().hex
        request_path = self.directory / "requests" / f"{request_id}.json"
        request_path.parent.mkdir(parents=True, exist_ok=True)
        if memory_selection is not None:
            selected = memory_selection.get("selected", [])
            relevant = sum(row["reason"] == "relevance" for row in selected)
            historical = sum(row["historical"] for row in selected)
            self.emit(
                "MEMORY_SEARCH",
                f"Просмотрено актуальных записей: {memory_selection.get('considered', 0)}; выбрано {len(selected)} (по теме {relevant}, недавние {len(selected) - relevant}, исторические {historical}); {memory_selection.get('chars', 0)}/{memory_selection.get('max_chars', 0)} символов.",
                text="\n".join(
                    f"{row['memory_id']} v{row['revision']} | {row['state']} | {row['reason']} | основы: {', '.join(row['matched_terms']) or 'нет'}"
                    for row in selected[:5]
                )
                + (
                    f"\nЕщё {len(selected)-5} записей — в файле запроса; /memory — полный поиск"
                    if len(selected) > 5
                    else ""
                ),
                selection=memory_selection,
                request_path=str(request_path),
            )
        request_path.write_text(
            json.dumps(
                {
                    "id": request_id,
                    "run_id": self.run_id,
                    "at": now.isoformat(),
                    "kind": kind,
                    "model": self.model_info,
                    "messages": messages,
                    "status": "pending",
                    "memory_selection": memory_selection,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        self.last_request = request_path
        self.inflight = PendingRequest(
            decision,
            request_path,
            time.monotonic(),
            time.monotonic(),
            diary_batch=diary_batch,
            life_date=life_date,
        )
        self.generations += 1
        self.emit(
            "REQUEST",
            f"Запрос #{self.generations}: {kind}. "
            f"Контекст: {len(messages)} сообщений, {sum(len(str(m['content'])) for m in messages)} символов; "
            "Точный состав контекста сохранён в файле запроса. Ожидаем ответ API.",
            request_path=str(request_path),
            scheduler_reason=(
                decision.reason
                if decision
                else (
                    "План смоделированного дня"
                    if life_date
                    else "Обновление дневника по новым входящим"
                )
            ),
        )

        def work() -> None:
            try:
                result = ModelResult(raw=self.bird.model.complete(messages))
            except Exception as exc:
                status = getattr(exc, "status_code", None)
                result = ModelResult(
                    error=type(exc).__name__ + (f" (HTTP {status})" if status else "")
                )
            self._results.put(result)

        threading.Thread(target=work, daemon=True, name="mockingbird-model").start()

    def _finish(self, result: ModelResult, now: datetime) -> None:
        pending = self.inflight
        if pending is None:
            return
        self.inflight = None
        elapsed = round(time.monotonic() - pending.started, 2)
        self._update_request(
            pending,
            response_raw=result.raw,
            error=result.error,
            elapsed_seconds=elapsed,
        )
        if (
            self.paused
            or pending.cancelled
            or (
                pending.decision is not None
                and not self.bird.scheduler.is_pending(pending.decision)
            )
        ):
            self._update_request(pending, status="discarded")
            self.emit(
                "DISCARD",
                "Результат не использован: запрос был приостановлен или контекст/план уже изменился.",
                request_path=str(pending.path),
            )
            return
        if pending.diary_batch is not None:
            self._finish_diary(pending, result, now)
            return
        if pending.life_date is not None:
            self._finish_life(pending, result, now)
            return
        assert pending.decision is not None
        changing_plan = False
        try:
            if result.error:
                raise RuntimeError(result.error)
            draft = self.bird.parse_draft(result.raw, pending.decision.kind)
            changing_plan = draft.life_update is not None
            if self.delivery is not None and draft.send:
                life_change = self.delivery(pending.decision, draft, now, pending.path)
            else:
                life_change = self.bird.commit(pending.decision, draft, now)
        except DeliveryDeferred as exc:
            self.delivery_deferred = True
            self._update_request(pending, status="delivery_deferred", error=str(exc))
            self.emit("DELIVERY_DEFERRED", str(exc), request_path=str(pending.path))
            return
        except DeliverySuperseded:
            self._update_request(
                pending, status="discarded", error="telegram_context_changed"
            )
            self.emit(
                "DISCARD",
                "Перед отправкой пришли новые сообщения. Соберём ответ заново.",
            )
            return
        except DeliveryBlocked as exc:
            self.paused = True
            self.delivery_blocked = True
            if self.state_sink is not None:
                self.state_sink()
            self._update_request(pending, status="delivery_blocked", error=str(exc))
            self.logger.error(
                "Delivery blocked request=%s reason=%s", pending.path.stem, exc
            )
            self.emit("DELIVERY_UNKNOWN", str(exc), request_path=str(pending.path))
            return
        except Exception as exc:
            self.logger.exception("Reply rejected request=%s", pending.path.stem)
            self.failures += 1
            self._update_request(
                pending,
                status="error",
                error=result.error or type(exc).__name__,
                validation_error=str(exc) if isinstance(exc, ValueError) else "",
            )
            self.bird.scheduler.retry_later(pending.decision, now)
            self.persist()
            if changing_plan:
                self.emit(
                    "LIFE_CHANGE_REJECTED",
                    "Правка плана не принята; согласие не отправлено. Старый план сохранён.",
                    reason=(
                        str(exc)
                        if isinstance(exc, ValueError)
                        else f"Не удалось сохранить правку ({type(exc).__name__})"
                    ),
                    request_path=str(pending.path),
                )
            self.emit(
                "ERROR",
                f"Ответ не принят: {result.error or type(exc).__name__}. Лимит отправленных не потрачен; подробности в файле запроса.",
                request_path=str(pending.path),
            )
            if self.failures >= self.max_failures:
                self.paused = True
                self.emit(
                    "PAUSED",
                    f"После {self.failures} ошибок подряд автоматические запросы остановлены. Проверь настройки и используй /resume.",
                )
            else:
                self.emit(
                    "RETRY",
                    "Повтор запланирован не ранее чем через пять минут, с учётом режима ответа и активных часов.",
                )
            return
        self.failures = 0
        self.persist()
        self._update_request(
            pending,
            status="sent" if draft.send else "model_skip",
            decision=asdict(draft),
            life_change=life_change,
        )
        if life_change is not None:
            assert self.bird.life is not None
            self.emit(
                "LIFE_CHANGED",
                self.bird.life.describe_change(life_change),
                reason=life_change["reason"],
                change=life_change,
                request_path=str(pending.path),
            )
        if draft.format == "plain_reply":
            self.emit(
                "FORMAT_FALLBACK",
                "Модель вернула готовую реплику без JSON. Ответ принят как текст; пояснение решения модель не предоставила.",
                request_path=str(pending.path),
            )
        self.emit(
            "OUTPUT" if draft.send else "MODEL_SKIP",
            f"{self.bird.history.user} {'написал' if draft.send else 'пропустил инициативу'} ({elapsed} сек.).",
            text=draft.text,
            **({"reason": draft.reason} if draft.reason else {}),
            kind=pending.decision.kind,
            request_path=str(pending.path),
        )

    def _finish_diary(
        self, pending: PendingRequest, result: ModelResult, now: datetime
    ) -> None:
        memory = self.bird.memory
        assert memory is not None and pending.diary_batch is not None
        try:
            if result.error:
                raise RuntimeError(result.error)
            update = memory.apply(pending.diary_batch, result.raw, now)
        except Exception as exc:
            self.failures += 1
            memory.defer(now)
            self._update_request(
                pending, status="error", error=result.error or type(exc).__name__
            )
            self.emit(
                "DIARY_ERROR",
                f"Дневник не обновлён: {result.error or type(exc).__name__}. Входящие сохранены для повторной попытки.",
                request_path=str(pending.path),
            )
            if self.failures >= self.max_failures:
                self.paused = True
                self.emit(
                    "PAUSED",
                    "Достигнут лимит ошибок. Проверь файл запроса и используй /resume.",
                )
            return
        self.failures = 0
        self.bird.diary = memory.combined()
        self._update_request(
            pending,
            status=(
                "diary_updated"
                if update.added + update.updated + update.confirmed + update.disputed
                else "diary_skip"
            ),
            decision=asdict(update),
        )
        stages = {
            "add": "DIARY_ADDED",
            "update": "DIARY_UPDATED",
            "confirm": "DIARY_CONFIRMED",
            "dispute": "DIARY_CONFLICT",
            "skip": "DIARY_SKIP",
        }
        for change in update.changes:
            latest = change.get("after")
            self.emit(
                stages[change["action"]],
                f"{change.get('target_id') or 'Без записи'}"
                + (f" v{latest['revision']} | {latest['state']}" if latest else ""),
                reason=change["reason"],
                reason_source=change.get("reason_scope", "operation"),
                text=latest["content"] if latest else "",
                source_ids=change["source_ids"],
                request_path=str(pending.path),
            )
        self.emit(
            "DIARY_RESULT",
            f"Добавлено: {update.added}; уточнено: {update.updated}; подтверждено: {update.confirmed}; спорно: {update.disputed}; пропущено: {update.skipped}. В очереди: {len(memory.pending)}.",
            reason=update.reason,
            reason_source="batch",
            request_path=str(pending.path),
        )

    def _finish_life(
        self, pending: PendingRequest, result: ModelResult, now: datetime
    ) -> None:
        life = self.bird.life
        assert life is not None and pending.life_date is not None
        if pending.life_date != life.local(now).date().isoformat():
            self._update_request(pending, status="discarded", error="life_date_changed")
            self.emit(
                "DISCARD",
                "Дата сменилась во время планирования дня. Составим план на текущую дату.",
            )
            return
        try:
            if result.error:
                raise RuntimeError(result.error)
            day = life.apply(result.raw, pending.life_date, now)
        except Exception as exc:
            life.defer(now)
            self.failures += 1
            self._update_request(
                pending, status="error", error=result.error or type(exc).__name__
            )
            self.emit(
                "LIFE_ERROR",
                f"План дня не принят: {result.error or type(exc).__name__}. Повтор не ранее чем через пять минут; события не выдуманы из ошибочного ответа.",
                request_path=str(pending.path),
            )
            if self.failures >= self.max_failures:
                self.paused = True
                self.emit("PAUSED", "Достигнут лимит ошибок. /resume — продолжить.")
            return
        self.failures = 0
        self._update_request(pending, status="life_planned", decision=day)
        self.emit(
            "LIFE_PLAN",
            life.view(now),
            request_path=str(pending.path),
        )

    def step(self) -> None:
        if self.closed or self.delivery_blocked or self.delivery_deferred:
            return
        now = self.clock()
        try:
            result = self._results.get_nowait()
        except queue.Empty:
            pass
        else:
            self._finish(result, now)
        if self.delivery_blocked or self.delivery_deferred:
            return
        self._advance_life()
        decision = self.bird.scheduler.tick(now)
        scheduler = self.bird.scheduler
        if self._shown_plan != scheduler.day:
            self._shown_plan = scheduler.day
            self.emit(
                "PLAN",
                f"План на {scheduler.day}: "
                + (
                    ", ".join(t.isoformat() for t in scheduler.planned)
                    or "инициативы не запланированы"
                )
                + f". Максимум {scheduler.config.max_initiations_per_day} в день.",
            )
        self.persist()
        if decision.action == Action.SKIP:
            self.emit("SKIP", decision.reason)
        if self.inflight is not None:
            if time.monotonic() - self.inflight.heartbeat >= 5:
                self.inflight.heartbeat = time.monotonic()
                self.emit(
                    "API_WAIT",
                    f"API ещё отвечает ({int(time.monotonic() - self.inflight.started)} сек.). Ввод и /status доступны.",
                )
            return
        if (
            self.bird.life is not None
            and self.bird.life.needed(now)
            and not self.paused
        ):
            if (
                self.max_generations is not None
                and self.generations >= self.max_generations
            ):
                self.paused = True
                self.emit(
                    "LIMIT",
                    "Лимит запросов запуска исчерпан; план дня пока не составлен.",
                )
            else:
                self._begin_life(now)
            return
        if decision.action == Action.SEND and not self.paused:
            if (
                self.max_generations is not None
                and self.generations >= self.max_generations
            ):
                self.paused = True
                self.emit(
                    "LIMIT",
                    f"Лимит {self.max_generations} запросов этого запуска исчерпан. Новых запросов не будет.",
                )
            else:
                self._begin(decision, now)
            return
        memory = self.bird.memory
        idle = (
            scheduler.last_incoming is None
            or (now - scheduler.last_incoming).total_seconds() >= 30
        )
        if (
            memory is not None
            and not self.paused
            and scheduler.reply_due is None
            and memory.due(now, force=self.diary_requested)
            and (idle or self.diary_requested)
        ):
            if (
                self.max_generations is not None
                and self.generations >= self.max_generations
            ):
                self.paused = True
                self.emit(
                    "LIMIT",
                    "Лимит запросов запуска исчерпан; очередь дневника сохранена.",
                )
            else:
                self._begin_diary(now)
            return
        wait = (decision.action, decision.due_at, self.paused, scheduler.awaiting_reply)
        if wait != self._shown_wait:
            self._shown_wait = wait
            if self.paused:
                message = "Запросы на паузе. Входящие сохраняются; /status — состояние."
            elif decision.due_at:
                message = f"Следующее событие: {decision.due_at.isoformat()}. Можно писать; /status — состояние."
            else:
                message = "Ожидаем твоё сообщение или план нового дня. Сегодня самостоятельных сообщений больше нет."
            self.emit("WAIT", message)

    def close(self, *, checkpoint_life: bool = True) -> None:
        if self.closed:
            return
        self.closed = True
        if checkpoint_life:
            self._advance_life(checkpoint=True)
        self.persist()
        if self.inflight is not None:
            self._update_request(
                self.inflight, status="abandoned", closed_at=self.clock().isoformat()
            )
        self.emit(
            "STOP",
            "Состояние сохранено, чат остановлен."
            + (
                " Выполняющийся HTTP-запрос может завершиться у провайдера, но его результат здесь уже не будет применён."
                if self.inflight
                else ""
            ),
        )
        for handler in list(self.logger.handlers):
            handler.close()
            self.logger.removeHandler(handler)
