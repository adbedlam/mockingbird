"""Local operator commands for the official Business Bot API transport."""

from __future__ import annotations

import asyncio
import json
import os
import queue
import threading
import time
from contextlib import ExitStack
from pathlib import Path

from mockingbird.chat.session import create_bird, create_pipeline
from mockingbird.config import ENV_PATH
from mockingbird.conversation.identity import pair_id
from mockingbird.interfaces.common import SessionLock, render
from mockingbird.interfaces.telegram.business import (
    BotAPIError,
    BotNetworkError,
    BusinessBotAPI,
    BusinessJournal,
    TelegramBusinessTransport,
)
from mockingbird.interfaces.telegram.options import TelegramOptions
from mockingbird.interfaces.telegram.state import SetupError, participant_id
from mockingbird.observability import export_bundle


def ready(transport: TelegramBusinessTransport) -> None:
    if not transport.send_enabled():
        raise SetupError("PERSONA_SEND_ENABLED=false: отправки отключены.")
    receipt = transport.journal.get("test_receipt")
    verified = transport.journal.get("appearance_verified")
    connection = transport.journal.get("connection", {})
    if (
        not receipt
        or not verified
        or verified.get("message_id") != receipt.get("message_id")
        or verified.get("connection_id") != connection.get("id")
    ):
        raise SetupError(
            "Сначала выполни подтверждённый тест и confirm-appearance для текущего подключения."
        )
    if transport.journal.get("review_required"):
        raise SetupError(
            "Есть изменение/удаление сообщения: сначала сверь историю и память. /review — детали."
        )
    if transport.journal.unresolved():
        raise SetupError("Есть отправка с неизвестным результатом. /delivery — детали.")
    if transport.journal.get("retry_after", 0) > time.time():
        raise SetupError("Ещё действует ограничение retry_after от Telegram.")


def run_persona(transport: TelegramBusinessTransport) -> None:
    assert transport.loop and transport.pipeline
    pipeline, loop = transport.pipeline, transport.loop
    commands: queue.Queue[str | None] = queue.Queue()

    def input_worker():
        while True:
            try:
                text = input()
            except EOFError:
                commands.put(None)
                return
            commands.put(text)
            if text.strip() == "/quit":
                return

    try:
        transport.recover()
        loop.run_until_complete(transport.restore_messages())
        try:
            loop.run_until_complete(transport.poll())
            while transport.backlog and not transport.journal.unresolved():
                loop.run_until_complete(transport.poll())
        except Exception as exc:
            transport.poll_failed(exc)
        try:
            ready(transport)
        except SetupError as exc:
            pipeline.paused = True
            transport.emit("BUSINESS_SEND_DISABLED", str(exc))
        pipeline.start(run_step=transport.network_available)
        threading.Thread(
            target=input_worker, daemon=True, name="business-commands"
        ).start()
        next_poll = 0.0
        while True:
            while not commands.empty():
                text = commands.get_nowait()
                if text is None or text.strip() == "/quit":
                    return
                text = text.strip()
                try:
                    if text == "/resume":
                        ready(transport)
                        pipeline.delivery_blocked = False
                    if text in ("/delivery", "/review"):
                        data = (
                            transport.journal.unresolved()
                            if text == "/delivery"
                            else transport.journal.get("review_required")
                        )
                        transport.emit(
                            "BUSINESS_REVIEW",
                            "Состояние для ручной сверки. resolve-delivery выполняется отдельной CLI-командой при остановленном процессе.",
                            text=json.dumps(data, ensure_ascii=False, indent=2),
                        )
                    elif text.startswith("/"):
                        if text == "/export":
                            transport.journal.report()
                        pipeline.command(text)
                        if text == "/pause":
                            transport.journal.set("pause_reason", "operator")
                            transport.journal.set("auto_pause_reason", None)
                        elif text == "/resume" and not pipeline.paused:
                            transport.journal.set("pause_reason", None)
                        elif text == "/status":
                            transport.emit(
                                "BUSINESS_HEALTH",
                                "Связь с Telegram: "
                                + (
                                    "доступна"
                                    if transport.network_available
                                    else "переподключаемся"
                                ),
                                pause_reason=transport.journal.get("pause_reason"),
                                poll_health=transport.journal.get("poll_health"),
                            )
                    else:
                        transport.emit(
                            "HELP",
                            "Здесь команды /status /pause /resume /delivery /review /export /quit. Общение идёт через Business Connection.",
                        )
                    transport.checkpoint()
                except SetupError as exc:
                    transport.emit("COMMAND_ERROR", str(exc))
            if time.monotonic() >= max(next_poll, transport.poll_retry_at):
                try:
                    loop.run_until_complete(transport.poll())
                except Exception as exc:
                    transport.poll_failed(exc)
                    next_poll = transport.poll_retry_at
                else:
                    next_poll = time.monotonic() + 2
            if (
                transport.network_available
                and not transport.journal.unresolved()
                and not transport.backlog
            ):
                pipeline.step()
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        pipeline.close(checkpoint_life=not bool(transport.journal.unresolved()))
        transport.checkpoint()
        transport.journal.report()


class TelegramInterface:
    def __init__(self, options: TelegramOptions):
        self.options = options

    def run(self, config: dict) -> None:
        _run(self.options, config)


def _run(options: TelegramOptions, config: dict) -> None:
    try:
        settings = config.get("telegram_business", {})
        owner = participant_id(config["preprocessing"]["user_id"])
        peer = participant_id(config["preprocessing"]["other_id"])
        allowed = settings.get("allowed_chat_ids", [])
        if allowed and allowed != [peer]:
            raise SetupError(
                "Разрешённый чат должен совпадать с other_id выбранной персоны."
            )
        identity = pair_id(config["user"], config["other"])
        runtime = Path(config["paths"]["runtime"])
        directory = options.state_dir or runtime / "business" / identity
        memory = Path(config["paths"]["memory"]) / identity
        with ExitStack() as stack:
            stack.enter_context(SessionLock(directory))
            journal = BusinessJournal(directory, (owner, peer))
            stack.callback(journal.close)
            if options.command == "export":
                journal.report()
                print(export_bundle(directory, memory))
                return
            if options.command == "status" and not options.live:
                journal.report()
                print((directory / "business_status.json").read_text(encoding="utf-8"))
                return
            if options.command == "confirm-appearance":
                receipt = journal.get("test_receipt")
                if not receipt or receipt["message_id"] != options.message_id:
                    raise SetupError(
                        "ID не совпал с подтверждённой Telegram тестовой отправкой."
                    )
                journal.set("appearance_verified", {**receipt, "owner_confirmed": True})
                journal.report()
                print(
                    "Владелец подтвердил отображение отправителя. Можно запускать run при PERSONA_SEND_ENABLED=true."
                )
                return
            loop = asyncio.new_event_loop()
            stack.callback(loop.close)
            api = BusinessBotAPI(os.getenv("TELEGRAM_BOT_TOKEN", ""))
            stack.callback(lambda: loop.run_until_complete(api.close()))
            transport = TelegramBusinessTransport(
                api,
                journal,
                owner_id=owner,
                allowed_chat_ids=allowed,
                bot_username=settings.get("bot_username", "mockingbird_persona_bot"),
                env_path=ENV_PATH,
                pause_on_manual_outgoing=settings.get("pause_on_manual_outgoing", True),
            )
            transport.loop = loop
            loop.run_until_complete(transport.initialize())
            if options.command == "status":
                loop.run_until_complete(transport.poll())
                journal.report()
                print((directory / "business_status.json").read_text(encoding="utf-8"))
                return
            if options.command == "observe":
                if options.seconds < 1 or options.seconds > 3600:
                    raise SetupError("Для observe задай --seconds от 1 до 3600.")
                end = time.monotonic() + options.seconds
                while time.monotonic() < end:
                    loop.run_until_complete(transport.poll())
                    time.sleep(min(2, max(0, end - time.monotonic())))
                journal.report()
                print("Наблюдение завершено. Модель и sendMessage не запускались.")
                return
            if options.command == "prepare-test":
                assert options.recipient is not None and options.text is not None
                loop.run_until_complete(transport.poll())
                print(
                    json.dumps(
                        transport.prepare_test(options.recipient, options.text),
                        ensure_ascii=False,
                        indent=2,
                    )
                )
                journal.report()
                return
            if options.command == "send-test":
                approval = journal.get("test_approval", {})
                if (
                    approval.get("id") != options.approval_id
                    or approval.get("text") != options.confirm_text
                ):
                    raise SetupError(
                        "Точное подтверждение не совпало с подготовленным тестом."
                    )
                approval["approved"] = True
                journal.set("test_approval", approval)
                transport._staged = (approval["id"], {"kind": "test", **approval})
                loop.run_until_complete(transport.poll())
                success = loop.run_until_complete(
                    transport.send_message(approval["recipient_id"], approval["text"])
                )
                journal.report()
                print(json.dumps(transport.last_delivery, ensure_ascii=False, indent=2))
                if not success:
                    raise SetupError("Тест не отправлен; смотри причину выше.")
                return
            stack.enter_context(SessionLock(memory))
            mockingbird = create_bird(
                config,
                fast_replies=(
                    options.fast_replies if options.command == "run" else False
                ),
            )
            if journal.get("state"):
                mockingbird.runtime_store.load_snapshot(journal.get("state"))
            else:
                mockingbird.runtime_store.restore(directory)
            pipeline = create_pipeline(
                config,
                mockingbird,
                directory,
                render,
                delivery=transport.deliver,
                state_sink=transport.checkpoint,
                max_generations=options.max_generations,
            )
            transport.pipeline = pipeline
            pipeline.paused = journal.get("paused", True)
            if options.command == "resolve-delivery":
                assert options.request_id is not None
                row = journal.row(options.request_id)
                if not row or row["status"] not in (
                    "prepared",
                    "sending",
                    "unknown",
                    "sent",
                ):
                    raise SetupError("Нет такой незавершённой отправки.")
                if options.outcome == "not-sent":
                    if row["status"] == "sent":
                        raise SetupError(
                            "Telegram уже подтвердил отправку; not-sent запрещён."
                        )
                    journal.update(
                        options.request_id, "rejected", error="owner_verified_not_sent"
                    )
                else:
                    if not options.message_id or options.message_id < 1:
                        raise SetupError(
                            "Нужен message-id, вручную сверенный владельцем в Telegram."
                        )
                    journal.update(
                        options.request_id, "sent", message_id=options.message_id
                    )
                    transport.recover()
                transport.checkpoint()
                journal.report()
                pipeline.close(checkpoint_life=not bool(journal.unresolved()))
                print("Ручная сверка сохранена. Повторной отправки не было.")
                return
            run_persona(transport)
    except SetupError as exc:
        raise SystemExit(f"Business: {exc}") from exc
    except BotAPIError as exc:
        raise SystemExit(f"Telegram Bot API {exc.code}: {exc}") from exc
    except BotNetworkError as exc:
        raise SystemExit(
            f"Telegram недоступен: {exc}. Проверь сеть/VPN и повтори запуск; отправки не выполнялись."
        ) from exc
    except KeyboardInterrupt:
        raise SystemExit(130) from None
    except Exception as exc:
        # Never stringify httpx exceptions: their URL contains the bot token.
        raise SystemExit(
            f"Business остановлен: {type(exc).__name__}. Проверь локальные логи и подключение."
        ) from None
