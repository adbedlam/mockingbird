from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import asdict
from datetime import date, datetime
from pathlib import Path
from typing import assert_never

from mockingbird.chat.errors import (
    DeliveryBlocked,
    DeliveryDeferred,
    DeliverySuperseded,
)
from mockingbird.chat.event_recorder.event_recorder import EventRecorder, PipelineEvent
from mockingbird.chat.presenter import status_text, wait_message
from mockingbird.chat.requests import RequestStore
from mockingbird.chat.snapshot import SnapshotStore
from mockingbird.chat.tasks import (
    DiaryTask,
    LifePlanTask,
    PendingRequest,
    ReplyTask,
    RequestTask,
)
from mockingbird.llm.model_runner import ModelRunner
from mockingbird.mockingbird.draft import Draft, parse_draft
from mockingbird.mockingbird.mockingbird import Mockingbird
from mockingbird.observability import export_bundle, start_run
from mockingbird.scheduler.scheduler import Action, Decision


class CommandRouter:
    def __init__(self, context: Chat):
        self.context = context

    def handle(self, text: str) -> bool:
        command = text.strip()
        if command == "/quit":
            return False
        if command in ("/help", "/?"):
            self._handle_help()
        elif command == "/export":
            self._handle_export()
        elif command == "/status":
            self._handle_status()
        elif command == "/life":
            self._handle_life()
        elif command == "/memory" or command.startswith("/memory "):
            self._handle_memory(command)
        elif command == "/plan":
            self._handle_plan()
        elif command == "/context":
            self._handle_context()
        elif command == "/context full":
            self._handle_context_full()
        elif command == "/diary":
            self._handle_diary()
        elif command == "/diary update":
            self._handle_diary_update()
        elif command == "/history":
            self._handle_history()
        elif command == "/pause":
            self._handle_pause()
        elif command == "/resume":
            self._handle_resume()
        else:
            self._handle_unknown()
        return True

    def _handle_help(self) -> None:
        self.context.emit(
            "HELP",
            "/status — status; /plan — initiatives; /life — simulated activities; /memory [topic] — search local memory; /context — exact request file; /context full — full request; /history — recent messages; /diary — diary and simulated days; /diary update — update from new incoming messages; /pause — pause; /resume — resume; /quit — exit.",
        )

    def _handle_export(self) -> None:
        context = self.context
        memory_directory = (
            context.mockingbird.memory.directory
            if context.mockingbird.memory
            else (
                context.mockingbird.life.directory.parent
                if context.mockingbird.life
                else None
            )
        )
        path = export_bundle(context.directory, memory_directory)
        context.emit("EXPORT", f"Debug archive with private conversation: {path}")

    def _handle_status(self) -> None:
        self.context.emit("STATUS", self.context.status())

    def _handle_life(self) -> None:
        context = self.context
        life = context.mockingbird.life
        context.emit(
            "LIFE_STATE",
            (
                life.view(context.clock())
                if life
                else "Day simulation is disabled in settings."
            ),
        )

    def _handle_memory(self, command: str) -> None:
        context = self.context
        query = command[len("/memory") :].strip()
        selected = context.mockingbird.memory_query.select(
            context.clock(), query or None
        )
        context.emit(
            "MEMORY_SEARCH",
            f"Local search: selected {len(selected.hits)} of {selected.considered} current entries; {selected.chars}/{selected.max_chars} characters. Query: {selected.query or 'current conversation'}",
            text="\n\n".join(
                f"{hit.entry.to_string()}\nSelected: {hit.reason}; matched stems: {', '.join(hit.matched_terms) or 'none'}; historical: {'yes' if hit.historical else 'no'}"
                for hit in selected.hits
            )
            or "No matches or suitable recent context.",
            selection=selected.report(),
        )

    def _handle_plan(self) -> None:
        scheduler = self.context.mockingbird.scheduler
        self.context.emit(
            "PLAN",
            f"Date: {scheduler.day}. Remaining initiatives: "
            + (", ".join(t.isoformat() for t in scheduler.planned) or "none")
            + ". Reply: "
            + (
                scheduler.reply_due.isoformat()
                if scheduler.reply_due
                else "not expected"
            ),
        )

    def _handle_context(self) -> None:
        context = self.context

        if context.last_request is None:
            context.emit(
                "CONTEXT",
                "No requests in this folder yet; context will be saved before the first generation.",
            )
            return

        context.emit("CONTEXT", str(context.last_request))

    def _handle_context_full(self) -> None:
        context = self.context

        if context.last_request is None:
            context.emit(
                "CONTEXT",
                "No requests in this folder yet; context will be saved before the first generation.",
            )
            return

        value = context.last_request.read_text(encoding="utf-8")
        context.emit("CONTEXT", value)

    def _handle_diary(self) -> None:
        context = self.context
        bird = context.mockingbird

        if bird.life is not None:
            context.advance_life(checkpoint=True)
            context.emit("DAILY_DIARY", bird.life.diary_view())

        if bird.memory is None:
            context.emit(
                "DIARY",
                "Memory accumulation is disabled for this run. Source diary:\n"
                + str(bird.diary),
            )
            return

        bird.memory.render()

        context.emit(
            "DIARY",
            f"{len(bird.memory.pending)} new incoming messages queued. "
            f"Diary: {bird.memory.directory / 'diary.md'}.\n"
            + str(bird.memory.combined()),
        )

    def _handle_diary_update(self) -> None:
        context = self.context
        memory = context.mockingbird.memory

        if memory is None:
            context.emit("DIARY_SKIP", "Memory accumulation is disabled.")
        elif not memory.pending:
            context.emit(
                "DIARY_SKIP",
                "No new incoming messages for the diary; "
                "previously processed messages won't be sent to the model again.",
            )
        else:
            context.diary_requested = True
            context.emit(
                "DIARY_QUEUED",
                "Diary update requested. We'll finish replying first, "
                "then report added entries or why the update was skipped.",
            )

    def _handle_history(self) -> None:
        rows = [
            f"{message.timestamp.isoformat()} | {message.author}: {message.text or ''}"
            for message in self.context.mockingbird.live.messages[-10:]
        ]
        self.context.emit(
            "HISTORY",
            "\n".join(rows) if rows else "No messages in the current conversation yet.",
        )

    def _handle_pause(self) -> None:
        context = self.context
        context.paused = True
        if context.inflight is not None:
            context.inflight.cancelled = True
        context.emit(
            "PAUSED",
            "New requests are paused. The result of any in-flight request will also be discarded. Use /resume to continue.",
        )

    def _handle_resume(self) -> None:
        context = self.context
        if (
            context.max_generations is not None
            and context.generations >= context.max_generations
        ):
            context.emit(
                "LIMIT",
                "This run's request limit is exhausted. Restart with --max-generations to set a new limit.",
            )
            return
        context.paused = False
        context.failures = 0
        context.emit(
            "RESUMED",
            "Generation resumed. Overdue initiatives won't be caught up.",
        )

    def _handle_unknown(self) -> None:
        self.context.emit(
            "HELP",
            "Unknown command. Use /help to list commands. To send text beginning with a slash to the model, prefix it with //.",
        )


class Chat:
    def __init__(
        self,
        mockingbird: Mockingbird,
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
        self.mockingbird = mockingbird
        self.directory = directory
        self.delivery = delivery
        self.delivery_blocked = False
        self.delivery_deferred = False
        self.state_sink = state_sink
        self.run_id, self.logger = start_run(directory, model_info or {})
        self.model_info = model_info or {}
        self.clock = clock or (lambda: datetime.now(mockingbird.scheduler.profile.zone))
        self.recorder = EventRecorder(directory, self.run_id, on_event, self.clock)
        self.requests = RequestStore(directory)
        self.snapshots = SnapshotStore()
        self.model_runner = ModelRunner(mockingbird.model.complete)
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
        self.snapshots.persist(self.mockingbird, self.directory, self.state_sink)

    def start(self, *, run_step: bool = True) -> None:
        self._emit_session_info()
        self._emit_mode_info()
        self._emit_restored_info()
        self._emit_freshness_info()
        self._emit_diary_info()
        self._emit_life_info()

        if run_step:
            self.step()

    def _emit_session_info(self) -> None:
        bird = self.mockingbird

        self.emit(
            "SESSION",
            f"Persona: {bird.history.user}; conversation partner: {bird.history.other}. "
            f"Model: {self.model_info.get('model', type(bird.model).__name__)}. "
            f"Time zone: {bird.scheduler.config['timezone']}.",
        )

    def _emit_mode_info(self) -> None:
        scheduler = self.mockingbird.scheduler
        config = scheduler.config

        if config["reply_mode"] == "immediate":
            message = (
                f"Fast replies: {config['min_reply_delay_seconds']} seconds after "
                "the last message fragment, even outside active hours. "
                "Initiatives follow the usual rules."
            )
        else:
            message = (
                "Replies and initiatives follow the timestamps in the history. "
                "Long waits are shown explicitly; use --fast-replies for a "
                "typical test conversation."
            )

        self.emit("MODE", message)

    def _emit_restored_info(self) -> None:
        bird = self.mockingbird

        self.emit(
            "RESTORED",
            f"Loaded: {len(bird.history.messages)} archived messages, "
            f"{len(bird.diary.entries)} diary entries, "
            f"{len(bird.live.messages)} current conversation messages. "
            f"Transcript: {self.directory / 'chat_transcript.md'}.",
        )

    def _emit_freshness_info(self) -> None:
        bird = self.mockingbird

        profile = bird.persona_sources.get("profile", {})

        message = (
            f"Archive: {bird.source_info.get('first')} — "
            f"{bird.source_info.get('last')}. "
            f"Source: {bird.source_origin or 'provided directly'}. "
            f"Profile: {profile.get('last', 'source data date unknown; --refresh-persona will rebuild it')}. "
            f"Build method: {profile.get('method', 'model_or_legacy')}. "
            f"Recent entries cover {bird.memory_max_age_days} days. "
        )

        message += self._memory_search_description()

        message += (
            f"Memory search is {'enabled' if bird.memory_search.enabled else 'disabled'}; "
            f"selected memory budget: {bird.memory_search.max_chars} characters."
        )

        self.emit("FRESHNESS", message)

    def _memory_search_description(self) -> str:
        search = self.mockingbird.memory_search

        if not search.enabled:
            return "Only recent context is used. "

        if search.max_age_days is None:
            return "Search covers all saved memory; old matches are marked historical. "

        return f"Search age limit: {search.max_age_days} days. "

    def _emit_diary_info(self) -> None:
        memory = self.mockingbird.memory

        if memory is None:
            return

        self.emit(
            "DIARY",
            f"Current memory entries: {len(memory.entries)}; "
            f"incoming messages queued: {len(memory.pending)}/{memory.update_every}. "
            f"Use /diary to view or /diary update to update manually. "
            f"File: {memory.directory / 'diary.md'}.",
        )

    def _emit_life_info(self) -> None:
        life = self.mockingbird.life

        if life is None:
            return

        self.emit(
            "LIFE",
            f"Simulated persona day: one plan per date. "
            f"/life shows activities; /diary shows the diary. "
            f"Summary at {life.diary_hour}:00 local time and on exit. "
            f"File: {life.directory / 'daily_diary.md'}.",
        )

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
        self.mockingbird.receive(text, at, message_id=message_id, timestamp=timestamp)
        self.persist()
        self.emit(
            "INPUT",
            f"Message received from {self.mockingbird.history.other}.",
            text=text.strip(),
            message_id=message_id,
            source_timestamp=timestamp.isoformat() if timestamp else None,
        )
        due = self.mockingbird.scheduler.reply_due
        self.emit(
            "SCHEDULE",
            f"Reply scheduled for {due.isoformat() if due else '—'}. "
            "New message fragments received before then will be combined into one reply.",
        )
        if self.inflight is not None and isinstance(self.inflight.task, ReplyTask):
            self.emit(
                "SUPERSEDED",
                "A new message arrived during generation. The old result will be discarded, then context will be rebuilt.",
            )

    def status(self) -> str:
        return status_text(
            self.mockingbird,
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
        messages = self.mockingbird.context_builder.build(decision, now)
        self._launch(
            messages,
            now,
            task=ReplyTask(decision),
            memory_selection=self.mockingbird.last_memory_selection,
        )

    def _begin_diary(self, now: datetime) -> None:
        memory = self.mockingbird.memory
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
        life = self.mockingbird.life
        assert life is not None
        self._launch(
            life.prepare(now, self.mockingbird.inner_prompt.content),
            now,
            task=LifePlanTask(life.local(now).date().isoformat()),
        )

    def advance_life(self, *, checkpoint: bool = False) -> None:
        if self.mockingbird.life is not None:
            for stage, message in self.mockingbird.life.advance(
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
            scheduler_reason = "Update diary from new incoming messages"
        elif isinstance(task, LifePlanTask):
            kind = "life_plan"
            scheduler_reason = "Plan simulated day"
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
                f"Current entries reviewed: {memory_selection.get('considered', 0)}; selected {len(selected)} (relevant {relevant}, recent {len(selected) - relevant}, historical {historical}); {memory_selection.get('chars', 0)}/{memory_selection.get('max_chars', 0)} characters.",
                text="\n".join(
                    f"{row['memory_id']} v{row['revision']} | {row['state']} | {row['reason']} | matched stems: {', '.join(row['matched_terms']) or 'none'}"
                    for row in selected[:5]
                )
                + (
                    f"\n{len(selected)-5} more entries are in the request file; use /memory for full search"
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
            f"Request #{self.generations}: {kind}. "
            f"Context: {len(messages)} messages, {sum(len(str(m['content'])) for m in messages)} characters; "
            "The exact context is saved in the request file. Waiting for the API response.",
            request_path=str(request_path),
            scheduler_reason=scheduler_reason,
        )

    def _take_inflight(self) -> PendingRequest | None:
        pending = self.inflight
        self.inflight = None
        return pending

    def _record_result(self, pending: PendingRequest, result: str) -> None:
        self._update_request(
            pending,
            response_raw=result,
            error=None,
            elapsed_seconds=round(time.monotonic() - pending.started, 2),
        )

    def _is_stale(self, pending: PendingRequest) -> bool:
        if self.paused or pending.cancelled:
            return True
        return isinstance(
            pending.task, ReplyTask
        ) and not self.mockingbird.scheduler.is_pending(pending.task.decision)

    def _discard(
        self, pending: PendingRequest, message: str, *, error: str | None = None
    ) -> None:
        if error is not None:
            self._update_request(pending, status="discarded", error=error)
        else:
            self._update_request(pending, status="discarded")
        self.emit("DISCARD", message, request_path=str(pending.path))

    def _handle_delivery_deferred(
        self, pending: PendingRequest, exc: DeliveryDeferred
    ) -> None:
        self.delivery_deferred = True
        self._update_request(pending, status="delivery_deferred", error=str(exc))
        self.emit("DELIVERY_DEFERRED", str(exc), request_path=str(pending.path))

    def _handle_delivery_blocked(
        self, pending: PendingRequest, exc: DeliveryBlocked
    ) -> None:
        self.paused = True
        self.delivery_blocked = True
        if self.state_sink is not None:
            self.state_sink()
        self._update_request(pending, status="delivery_blocked", error=str(exc))
        self.logger.error(
            "Delivery blocked request=%s reason=%s", pending.path.stem, exc
        )
        self.emit("DELIVERY_UNKNOWN", str(exc), request_path=str(pending.path))

    def _handle_reply_error(
        self,
        pending: PendingRequest,
        decision: Decision,
        exc: Exception,
        changing_plan: bool,
        now: datetime,
    ) -> None:
        error = str(exc) or type(exc).__name__
        self.logger.exception("Reply rejected request=%s", pending.path.stem)
        self.failures += 1
        self._update_request(
            pending,
            status="error",
            error=error,
            validation_error=str(exc) if isinstance(exc, ValueError) else "",
        )
        self.mockingbird.scheduler.retry_later(decision, now)
        self.persist()
        if changing_plan:
            self.emit(
                "LIFE_CHANGE_REJECTED",
                "Plan change rejected; no approval was sent. The previous plan was kept.",
                reason=(
                    str(exc)
                    if isinstance(exc, ValueError)
                    else f"Could not save the plan change ({type(exc).__name__})"
                ),
                request_path=str(pending.path),
            )
        self.emit(
            "ERROR",
            f"Reply rejected: {error}. The send limit was not used; see the request file for details.",
            request_path=str(pending.path),
        )
        if self.failures >= self.max_failures:
            self.paused = True
            self.emit(
                "PAUSED",
                f"Automatic requests paused after {self.failures} consecutive errors. Check the settings and use /resume.",
            )
        else:
            self.emit(
                "RETRY",
                "A retry is scheduled no sooner than five minutes from now, subject to the reply mode and active hours.",
            )

    def _handle_reply_success(
        self,
        pending: PendingRequest,
        decision: Decision,
        draft: Draft,
        life_change: dict | None,
    ) -> None:
        elapsed = round(time.monotonic() - pending.started, 2)
        self.failures = 0
        self.persist()
        self._update_request(
            pending,
            status="sent" if draft.send else "model_skip",
            decision=asdict(draft),
            life_change=life_change,
        )
        if life_change is not None:
            assert self.mockingbird.life is not None
            self.emit(
                "LIFE_CHANGED",
                self.mockingbird.life.describe_change(life_change),
                reason=life_change["reason"],
                change=life_change,
                request_path=str(pending.path),
            )
        if draft.format == "plain_reply":
            self.emit(
                "FORMAT_FALLBACK",
                "The model returned a complete reply without JSON. The text was accepted, but the model provided no explanation.",
                request_path=str(pending.path),
            )
        self.emit(
            "OUTPUT" if draft.send else "MODEL_SKIP",
            f"{self.mockingbird.history.user} {'sent a reply' if draft.send else 'skipped an initiative'} ({elapsed} sec.).",
            text=draft.text,
            **({"reason": draft.reason} if draft.reason else {}),
            kind=decision.kind,
            request_path=str(pending.path),
        )

    def _finish_reply(
        self,
        pending: PendingRequest,
        task: ReplyTask,
        draft: Draft,
        now: datetime,
    ) -> dict | None:
        decision = task.decision
        if self.delivery is not None and draft.send:
            return self.delivery(decision, draft, now, pending.path)
        return self.mockingbird.commit(decision, draft, now)

    def _handle_diary_error(
        self, pending: PendingRequest, exc: Exception, now: datetime
    ) -> None:
        memory = self.mockingbird.memory
        assert memory is not None
        error = str(exc) or type(exc).__name__
        self.failures += 1
        memory.defer(now)
        self._update_request(pending, status="error", error=error)
        self.emit(
            "DIARY_ERROR",
            f"Diary not updated: {error}. Incoming messages were saved for a retry.",
            request_path=str(pending.path),
        )
        if self.failures >= self.max_failures:
            self.paused = True
            self.emit(
                "PAUSED",
                "Error limit reached. Check the request file and use /resume.",
            )

    def _handle_life_error(
        self, pending: PendingRequest, exc: Exception, now: datetime
    ) -> None:
        life = self.mockingbird.life
        assert life is not None
        error = str(exc) or type(exc).__name__
        life.defer(now)
        self.failures += 1
        self._update_request(pending, status="error", error=error)
        self.emit(
            "LIFE_ERROR",
            f"Daily plan rejected: {error}. Retry in no less than five minutes; no events were invented from the invalid response.",
            request_path=str(pending.path),
        )
        if self.failures >= self.max_failures:
            self.paused = True
            self.emit("PAUSED", "Error limit reached. Use /resume to continue.")

    def _finish_model_error(self, exc: Exception, now: datetime) -> None:
        pending = self._take_inflight()
        if pending is None:
            return
        self._update_request(
            pending,
            error=str(exc) or type(exc).__name__,
            elapsed_seconds=round(time.monotonic() - pending.started, 2),
        )
        if self._is_stale(pending):
            self._discard(
                pending,
                "Result discarded: the request was paused or the context/plan changed.",
            )
            return
        task = pending.task
        if isinstance(task, ReplyTask):
            self._handle_reply_error(pending, task.decision, exc, False, now)
        elif isinstance(task, DiaryTask):
            self._handle_diary_error(pending, exc, now)
        else:
            assert isinstance(task, LifePlanTask)
            life = self.mockingbird.life
            assert life is not None
            if task.date != life.local(now).date().isoformat():
                self._update_request(
                    pending, status="discarded", error="life_date_changed"
                )
                self.emit(
                    "DISCARD",
                    "The date changed while planning the day. A plan will be created for the current date.",
                )
                return
            self._handle_life_error(pending, exc, now)

    def _finish(self, result: str, now: datetime) -> None:
        pending = self._take_inflight()
        if pending is None:
            return
        self._record_result(pending, result)
        if self._is_stale(pending):
            self._discard(
                pending,
                "Result discarded: the request was paused or the context/plan changed.",
            )
            return
        task = pending.task
        if isinstance(task, DiaryTask):
            self._finish_diary(pending, task, result, now)
        elif isinstance(task, LifePlanTask):
            self._finish_life(pending, task, result, now)
        else:
            assert isinstance(task, ReplyTask)
            changing_plan = False
            try:
                draft = parse_draft(result, task.decision.kind)
                changing_plan = draft.life_update is not None
                life_change = self._finish_reply(pending, task, draft, now)
            except DeliveryDeferred as exc:
                self._handle_delivery_deferred(pending, exc)
                return
            except DeliverySuperseded:
                self._discard(
                    pending,
                    "New messages recieved before sending.",
                    error="telegram_context_changed",
                )
                return
            except DeliveryBlocked as exc:
                self._handle_delivery_blocked(pending, exc)
                return
            except Exception as exc:
                self._handle_reply_error(
                    pending, task.decision, exc, changing_plan, now
                )
                return
            self._handle_reply_success(pending, task.decision, draft, life_change)

    def _finish_diary(
        self, pending: PendingRequest, task: DiaryTask, result: str, now: datetime
    ) -> None:
        memory = self.mockingbird.memory
        assert memory is not None
        try:
            update = memory.apply(task.batch, result, now)
        except Exception as exc:
            self._handle_diary_error(pending, exc, now)
            return
        self.failures = 0
        self.mockingbird.diary = memory.combined()
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
                f"{change.get('target_id') or 'No entry'}"
                + (f" v{latest['revision']} | {latest['state']}" if latest else ""),
                reason=change["reason"],
                reason_source=change.get("reason_scope", "operation"),
                text=latest["content"] if latest else "",
                source_ids=change["source_ids"],
                request_path=str(pending.path),
            )
        self.emit(
            "DIARY_RESULT",
            f"Added: {update.added}; updated: {update.updated}; confirmed: {update.confirmed}; disputed: {update.disputed}; skipped: {update.skipped}. Queued: {len(memory.pending)}.",
            reason=update.reason,
            reason_source="batch",
            request_path=str(pending.path),
        )

    def _finish_life(
        self,
        pending: PendingRequest,
        task: LifePlanTask,
        result: str,
        now: datetime,
    ) -> None:
        life = self.mockingbird.life
        assert life is not None
        if task.date != life.local(now).date().isoformat():
            self._update_request(pending, status="discarded", error="life_date_changed")
            self.emit(
                "DISCARD",
                "The date changed while planning the day. A plan will be created for the current date.",
            )
            return
        try:
            day = life.apply(result, task.date, now)
        except Exception as exc:
            self._handle_life_error(pending, exc, now)
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
            try:
                result = pending.future.result()
            except Exception as exc:
                self._finish_model_error(exc, now)
            else:
                self._finish(result, now)
        if self.delivery_blocked or self.delivery_deferred:
            return
        self.advance_life()
        decision = self.mockingbird.scheduler.tick(now)
        scheduler = self.mockingbird.scheduler
        if self._shown_plan != scheduler.day:
            self._shown_plan = scheduler.day
            self.emit(
                "PLAN",
                f"Plan for {scheduler.day}: "
                + (
                    ", ".join(t.isoformat() for t in scheduler.planned)
                    or "no initiatives scheduled"
                )
                + f". Maximum {scheduler.config['max_initiations_per_day']} per day.",
            )
        self.persist()
        if decision.action == Action.SKIP:
            self.emit("SKIP", decision.reason)
        if self.inflight is not None:
            if time.monotonic() - self.inflight.heartbeat >= 5:
                self.inflight.heartbeat = time.monotonic()
                self.emit(
                    "API_WAIT",
                    f"API is still responding ({int(time.monotonic() - self.inflight.started)} sec.). Input and /status are available.",
                )
            return
        if (
            self.mockingbird.life is not None
            and self.mockingbird.life.needed(now)
            and not self.paused
        ):
            if (
                self.max_generations is not None
                and self.generations >= self.max_generations
            ):
                self.paused = True
                self.emit(
                    "LIMIT",
                    "Run request limit exhausted; the daily plan has not been created.",
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
                    f"This run's limit of {self.max_generations} requests is exhausted. No more requests will be made.",
                )
            else:
                self._begin(decision, now)
            return
        memory = self.mockingbird.memory
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
                    "Run request limit exhausted; the diary queue has been preserved.",
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
            "State saved; chat stopped."
            + (
                " The in-flight HTTP request may finish at the provider, but its result will not be applied here."
                if self.inflight
                else ""
            ),
        )
        for handler in list(self.logger.handlers):
            handler.close()
            self.logger.removeHandler(handler)
