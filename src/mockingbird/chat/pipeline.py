"""Coordinate chat scheduling, model work, persistence, and delivery.

Only the main thread touches history and scheduler state. The model runner
returns raw output for the coordinator to validate and apply.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import asdict
from datetime import date, datetime
from pathlib import Path
from typing import assert_never

from mockingbird.chat.commands import CommandRouter
from mockingbird.chat.errors import (
    DeliveryBlocked,
    DeliveryDeferred,
    DeliverySuperseded,
)
from mockingbird.chat.events import EventRecorder, PipelineEvent
from mockingbird.chat.model_runner import ModelRunner
from mockingbird.chat.presenter import status_text, wait_message
from mockingbird.chat.requests import RequestStore
from mockingbird.chat.snapshot import SnapshotStore
from mockingbird.chat.tasks import (
    DiaryTask,
    LifePlanTask,
    ModelResult,
    PendingRequest,
    ReplyTask,
    RequestTask,
)
from mockingbird.mockingbird.mockingbird import Draft, Mockingbird
from mockingbird.observability import start_run
from mockingbird.scheduler.scheduler import Action, Decision


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
        self.model_info = model_info or {}
        self.clock = clock or (lambda: datetime.now(bird.scheduler.profile.zone))
        self.recorder = EventRecorder(directory, self.run_id, on_event, self.clock)
        self.requests = RequestStore(directory)
        self.snapshots = SnapshotStore()
        self.model_runner = ModelRunner(bird.model.complete)
        self.max_failures = max_failures
        self.max_generations = max_generations
        self.generations = 0
        self.failures = 0
        self.paused = False
        self.closed = False
        self.inflight: PendingRequest | None = None
        self._shown_plan: date | None = None
        self._shown_wait: tuple | None = None
        self.last_request = self.requests.latest()
        self.diary_requested = False
        self.command_router = CommandRouter(self)

    def emit(self, stage: str, message: str, **data: object) -> None:
        self.recorder.emit(stage, message, **data)

    def persist(self) -> None:
        self.snapshots.persist(self.bird, self.directory, self.state_sink)

    def start(self, *, run_step: bool = True) -> None:
        bird = self.bird
        self.emit(
            "SESSION",
            f"Персона: {bird.history.user}; собеседник: {bird.history.other}. "
            f"Модель: {self.model_info.get('model', type(bird.model).__name__)}. "
            f"Часовой пояс: {bird.scheduler.config['timezone']}.",
        )
        mode = bird.scheduler.config["reply_mode"]
        self.emit(
            "MODE",
            (
                f"Быстрые ответы: через {bird.scheduler.config['min_reply_delay_seconds']} сек. после последнего фрагмента, даже вне активных часов. Инициативы — по обычным правилам."
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
        if self.inflight is not None and isinstance(self.inflight.task, ReplyTask):
            self.emit(
                "SUPERSEDED",
                "Пришло новое сообщение во время генерации. Старый результат будет отброшен; затем соберём обновлённый контекст.",
            )

    def status(self) -> str:
        return status_text(
            self.bird,
            self.clock(),
            inflight=self.inflight is not None,
            generations=self.generations,
            max_generations=self.max_generations,
            paused=self.paused,
            failures=self.failures,
            max_failures=self.max_failures,
        )

    def command(self, text: str) -> bool:
        return self.command_router.handle(text)

    def _update_request(self, pending: PendingRequest, **changes: object) -> None:
        self.requests.update(pending.path, **changes)

    def _begin(self, decision: Decision, now: datetime) -> None:
        messages = self.bird.context(decision, now)
        self._launch(
            messages,
            now,
            task=ReplyTask(decision),
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
            task=DiaryTask(batch),
            memory_selection=batch.memory_selection,
        )

    def _begin_life(self, now: datetime) -> None:
        life = self.bird.life
        assert life is not None
        self._launch(
            life.prepare(now, self.bird.inner_prompt.content),
            now,
            task=LifePlanTask(life.local(now).date().isoformat()),
        )

    def advance_life(self, *, checkpoint: bool = False) -> None:
        if self.bird.life is not None:
            for stage, message in self.bird.life.advance(
                self.clock(), checkpoint=checkpoint
            ):
                self.emit(stage, message)

    def _launch(
        self,
        messages: list,
        now: datetime,
        *,
        task: RequestTask,
        memory_selection: dict | None = None,
    ) -> None:
        if isinstance(task, ReplyTask):
            kind = str(task.decision.kind)
            scheduler_reason = task.decision.reason
        elif isinstance(task, DiaryTask):
            kind = "diary"
            scheduler_reason = "Обновление дневника по новым входящим"
        elif isinstance(task, LifePlanTask):
            kind = "life_plan"
            scheduler_reason = "План смоделированного дня"
        else:
            assert_never(task)

        request_path = self.requests.create(
            run_id=self.run_id,
            at=now,
            kind=kind,
            model=self.model_info,
            messages=messages,
            memory_selection=memory_selection,
        )
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
        self.last_request = request_path
        started = time.monotonic()
        future = self.model_runner.submit(messages)
        self.inflight = PendingRequest(
            task,
            request_path,
            started,
            started,
            future,
        )
        self.generations += 1
        self.emit(
            "REQUEST",
            f"Запрос #{self.generations}: {kind}. "
            f"Контекст: {len(messages)} сообщений, {sum(len(str(m['content'])) for m in messages)} символов; "
            "Точный состав контекста сохранён в файле запроса. Ожидаем ответ API.",
            request_path=str(request_path),
            scheduler_reason=scheduler_reason,
        )

    def _finish(self, result: ModelResult, now: datetime) -> None:
        pending = self.inflight
        if pending is None:
            return
        self.inflight = None
        task = pending.task
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
                isinstance(task, ReplyTask)
                and not self.bird.scheduler.is_pending(task.decision)
            )
        ):
            self._update_request(pending, status="discarded")
            self.emit(
                "DISCARD",
                "Результат не использован: запрос был приостановлен или контекст/план уже изменился.",
                request_path=str(pending.path),
            )
            return
        if isinstance(task, DiaryTask):
            self._finish_diary(pending, task, result, now)
            return
        if isinstance(task, LifePlanTask):
            self._finish_life(pending, task, result, now)
            return
        assert isinstance(task, ReplyTask)
        decision = task.decision
        changing_plan = False
        try:
            if result.error:
                raise RuntimeError(result.error)
            draft = self.bird.parse_draft(result.raw, decision.kind)
            changing_plan = draft.life_update is not None
            if self.delivery is not None and draft.send:
                life_change = self.delivery(decision, draft, now, pending.path)
            else:
                life_change = self.bird.commit(decision, draft, now)
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
            self.bird.scheduler.retry_later(decision, now)
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
            kind=decision.kind,
            request_path=str(pending.path),
        )

    def _finish_diary(
        self,
        pending: PendingRequest,
        task: DiaryTask,
        result: ModelResult,
        now: datetime,
    ) -> None:
        memory = self.bird.memory
        assert memory is not None
        try:
            if result.error:
                raise RuntimeError(result.error)
            update = memory.apply(task.batch, result.raw, now)
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
        self,
        pending: PendingRequest,
        task: LifePlanTask,
        result: ModelResult,
        now: datetime,
    ) -> None:
        life = self.bird.life
        assert life is not None
        if task.date != life.local(now).date().isoformat():
            self._update_request(pending, status="discarded", error="life_date_changed")
            self.emit(
                "DISCARD",
                "Дата сменилась во время планирования дня. Составим план на текущую дату.",
            )
            return
        try:
            if result.error:
                raise RuntimeError(result.error)
            day = life.apply(result.raw, task.date, now)
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
        pending = self.inflight
        if pending is not None and pending.future.done():
            self._finish(pending.future.result(), now)
        if self.delivery_blocked or self.delivery_deferred:
            return
        self.advance_life()
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
                + f". Максимум {scheduler.config['max_initiations_per_day']} в день.",
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
            self.emit("WAIT", wait_message(decision, self.paused))

    def close(self, *, checkpoint_life: bool = True) -> None:
        if self.closed:
            return
        self.closed = True
        if checkpoint_life:
            self.advance_life(checkpoint=True)
        self.persist()
        if self.inflight is not None:
            self.requests.abandon(self.inflight.path, self.clock())
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
