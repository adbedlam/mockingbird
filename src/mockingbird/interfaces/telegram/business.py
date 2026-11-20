from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

from mockingbird.chat.errors import (
    DeliveryBlocked,
    DeliveryDeferred,
    DeliverySuperseded,
)
from mockingbird.chat.pipeline import ChatPipeline
from mockingbird.config import save_business_connection_id
from mockingbird.conversation.message import Message
from mockingbird.interfaces.telegram.state import DeliveryJournal, SetupError
from mockingbird.mockingbird.draft import Draft
from mockingbird.mockingbird.mockingbird import Mockingbird
from mockingbird.observability import redact
from mockingbird.scheduler.scheduler import Decision

BUSINESS_UPDATES = [
    "business_connection",
    "business_message",
    "edited_business_message",
    "deleted_business_messages",
]


class BotAPIError(RuntimeError):
    def __init__(self, code: int, description: str, retry_after: int = 0):
        self.code, self.retry_after = code, retry_after
        super().__init__(redact(description))


class BotNetworkError(RuntimeError):
    def __init__(self, detail: str, *, method: str = ""):
        self.detail, self.method = detail, method
        super().__init__(f"{method}: {detail}" if method else detail)


class BusinessBotAPI:
    def __init__(
        self, token: str, *, transport: httpx.AsyncBaseTransport | None = None
    ):
        if not token or ":" not in token:
            raise SetupError("Заполни TELEGRAM_BOT_TOKEN в локальном .env.")
        self._token = token
        # httpx INFO messages include the token-bearing Bot API URL.
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(25, connect=10),
            follow_redirects=False,
            transport=transport,
        )

    async def call(self, method: str, **params: Any) -> Any:
        if method not in {
            "getMe",
            "getWebhookInfo",
            "getUpdates",
            "getBusinessConnection",
            "sendMessage",
        }:
            raise ValueError("Unsupported Bot API method")
        try:
            response = await self.client.post(
                f"https://api.telegram.org/bot{self._token}/{method}", json=params
            )
        except httpx.TransportError as exc:
            raise BotNetworkError(type(exc).__name__, method=method) from None
        try:
            data = response.json()
        except ValueError:
            raise BotNetworkError(
                f"InvalidJSON (HTTP {response.status_code})", method=method
            ) from None
        if not isinstance(data, dict):
            raise BotNetworkError("Invalid Bot API response", method=method)
        if data.get("ok") is not True:
            description = str(
                data.get("description", "Bot API rejected request")
            ).replace(self._token, "[REDACTED]")
            raise BotAPIError(
                int(data.get("error_code", response.status_code)),
                description,
                int((data.get("parameters") or {}).get("retry_after", 0)),
            )
        return data.get("result")

    async def close(self) -> None:
        await self.client.aclose()


class BusinessJournal(DeliveryJournal):
    def __init__(self, directory: Path, identity: tuple[int, int]):
        super().__init__(directory, identity)
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS business_updates (id INTEGER PRIMARY KEY, payload TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'queued')"
        )
        self.db.commit()

    def queue_updates(self, updates: list[dict]) -> None:
        # Persist before acknowledging via getUpdates(offset). An interrupted
        # application can replay the local queue without fetching an event twice.
        with self.db:
            for update in updates:
                self.db.execute(
                    "INSERT OR IGNORE INTO business_updates(id,payload) VALUES (?,?)",
                    (update["update_id"], json.dumps(update, ensure_ascii=False)),
                )
            if updates:
                self._set(
                    "offset",
                    max(
                        self.get("offset", 0),
                        max(row["update_id"] for row in updates) + 1,
                    ),
                )

    def queued(self) -> list[dict]:
        return [
            json.loads(row[0])
            for row in self.db.execute(
                "SELECT payload FROM business_updates WHERE status='queued' ORDER BY id"
            )
        ]

    def applied(self, update_id: int, bird: Mockingbird | None, paused: bool) -> None:
        with self.db:
            if bird is not None:
                self._set("state", bird.runtime_store.snapshot())
            self._set("paused", paused)
            self.db.execute(
                "UPDATE business_updates SET status='applied' WHERE id=?", (update_id,)
            )

    def row(self, identity: str) -> dict | None:
        row = self.db.execute("SELECT * FROM outbox WHERE id=?", (identity,)).fetchone()
        return dict(row) if row else None

    def report(self) -> None:
        super().report()
        self.write_status()

    def write_status(self) -> None:
        data = {
            key: self.get(key)
            for key in (
                "bot",
                "connection",
                "last_incoming",
                "offset",
                "test_approval",
                "test_receipt",
                "appearance_verified",
                "review_required",
                "verified_recipient",
                "pause_reason",
                "poll_health",
                "poll_health",
                "pause_reason",
            )
        }
        data["queued_updates"] = len(self.queued())
        data["synced_at"] = datetime.now(timezone.utc).isoformat()
        data["paused"] = self.get("paused", True)
        data["recent_deliveries"] = [
            dict(row)
            for row in self.db.execute(
                "SELECT id,status,text,message_id,error,created_at FROM outbox ORDER BY created_at DESC LIMIT 50"
            )
        ]
        data["send_enabled"] = (
            os.getenv("PERSONA_SEND_ENABLED", "false").strip().lower() == "true"
        )
        path = self.directory / "business_status.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temporary.replace(path)


class TelegramBusinessTransport:
    def __init__(
        self,
        api: Any,
        journal: BusinessJournal,
        *,
        owner_id: int,
        allowed_chat_ids: list[int],
        bot_username: str = "mockingbird_persona_bot",
        clock: Any = None,
        send_enabled: Any = None,
        env_path: Path | None = None,
        pause_on_manual_outgoing: bool = True,
    ):
        if len(allowed_chat_ids) > 1 or any(
            type(v) is not int or v < 1 or v == owner_id for v in allowed_chat_ids
        ):
            raise SetupError(
                "Первая версия разрешает только один явно указанный личный чат, отличный от владельца."
            )
        self.api, self.journal = api, journal
        self.owner_id, self.allowed = owner_id, frozenset(allowed_chat_ids)
        self.bot_username = bot_username.lstrip("@").lower()
        self.env_path = env_path
        if type(pause_on_manual_outgoing) is not bool:
            raise SetupError(
                "telegram_business.pause_on_manual_outgoing must be true or false"
            )
        self.pause_on_manual_outgoing = pause_on_manual_outgoing
        self.network_available = True
        self.poll_retry_at = 0.0
        self.poll_failures = 0
        self._last_poll_error = ""
        self._last_poll_log = 0.0
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self.send_enabled = send_enabled or (
            lambda: os.getenv("PERSONA_SEND_ENABLED", "false").strip().lower() == "true"
        )
        self.pipeline: ChatPipeline | None = None
        self.loop: asyncio.AbstractEventLoop | None = None
        self.last_delivery: dict = {}
        self.backlog = False
        self._staged: tuple[str, dict] | None = None

    def emit(self, stage: str, message: str, **data: Any) -> None:
        self.journal.write_status()
        if self.pipeline:
            self.pipeline.emit(stage, message, **data)
        else:
            event = {
                "at": self.clock().isoformat(),
                "stage": stage,
                "message": message,
                "data": data,
            }
            with (self.journal.directory / "decisions.jsonl").open(
                "a", encoding="utf-8"
            ) as stream:
                stream.write(redact(json.dumps(event, ensure_ascii=False)) + "\n")
            print(f"{stage} | {message}", flush=True)

    async def initialize(self) -> dict:
        bot = await self.api.call("getMe")
        if (
            not bot.get("is_bot")
            or str(bot.get("username", "")).lower() != self.bot_username
        ):
            raise SetupError("Токен принадлежит другому боту.")
        if bot.get("can_connect_to_business") is not True:
            raise SetupError("У бота не включён Secretary / Business Mode.")
        previous = self.journal.get("bot")
        if previous and previous["id"] != bot["id"]:
            raise SetupError("Папка состояния принадлежит другому боту.")
        self.journal.set(
            "bot", {k: bot[k] for k in ("id", "username", "can_connect_to_business")}
        )
        webhook = await self.api.call("getWebhookInfo")
        if webhook.get("url"):
            raise SetupError(
                "У бота настроен webhook. Этот polling-процесс его не удаляет; сначала согласуй остановку прежнего обработчика."
            )
        configured = os.getenv("TELEGRAM_BUSINESS_CONNECTION_ID", "").strip()
        known_id = configured or (self.journal.get("connection") or {}).get("id")
        if known_id:
            try:
                await self.refresh_connection(known_id)
            except BotAPIError as exc:
                if exc.code not in (400, 404):
                    raise
                self.emit(
                    "BUSINESS_DISCOVERY",
                    "Сохранённое подключение недоступно. Ожидаем новое событие подключения владельца.",
                )
        return bot

    def store_connection(self, connection: dict) -> bool:
        if connection.get("user", {}).get("id") != self.owner_id:
            self.emit(
                "BUSINESS_IGNORED", "Подключение другого владельца проигнорировано."
            )
            return False
        previous = self.journal.get("connection") or {}
        if (
            previous.get("id") != connection.get("id")
            and previous.get("is_enabled")
            and (
                not connection.get("is_enabled")
                or connection.get("date", 0) < previous.get("date", 0)
            )
        ):
            return False
        self.journal.set("connection", connection)
        if self.env_path is not None and connection.get("is_enabled"):
            changed = save_business_connection_id(connection["id"], self.env_path)
            os.environ["TELEGRAM_BUSINESS_CONNECTION_ID"] = connection["id"]
            if changed:
                self.emit(
                    "BUSINESS_ENV_SAVED",
                    "Business Connection ID автоматически сохранён в локальном .env.",
                )
        self.emit(
            "BUSINESS_CONNECTION",
            "Подключение владельца сохранено.",
            connection_id=connection["id"],
            owner_id=self.owner_id,
            user_chat_id=connection.get("user_chat_id"),
            is_enabled=connection.get("is_enabled"),
            can_reply=connection.get("rights", {}).get("can_reply", False),
        )
        return True

    async def refresh_connection(self, identity: str) -> dict:
        connection = await self.api.call(
            "getBusinessConnection", business_connection_id=identity
        )
        if connection.get("id") != identity or not self.store_connection(connection):
            raise SetupError(
                "Business Connection не соответствует ожидаемому владельцу или ID."
            )
        return connection

    def checkpoint(self) -> None:
        if self.pipeline:
            self.journal.save_state(self.pipeline.bird, self.pipeline.paused)

    def pause(self, message: str, *, reason: str = "operator", **data: Any) -> None:
        self.journal.set("auto_pause_reason", None)
        previous = self.journal.get("pause_reason")
        self.journal.set("pause_reason", reason)
        was_paused = (
            self.pipeline.paused if self.pipeline else self.journal.get("paused", False)
        )
        if self.pipeline and not was_paused:
            self.pipeline.command("/pause")
        self.journal.set("paused", True)
        if not was_paused or previous != reason:
            self.emit("BUSINESS_PAUSED", message, pause_reason=reason, **data)

    @staticmethod
    def transient_read_error(exc: Exception) -> bool:
        return isinstance(exc, BotNetworkError) or (
            isinstance(exc, BotAPIError) and (exc.code >= 500 or exc.code == 429)
        )

    def poll_failed(self, exc: Exception) -> float:
        """Transport health never clears or creates an operator pause."""
        self.network_available = False
        if self.pipeline:
            self.pipeline.delivery_deferred = True
        self.poll_failures += 1
        delay = max(
            min(15 * 2 ** min(self.poll_failures - 1, 2), 60),
            getattr(exc, "retry_after", 0),
        )
        self.poll_retry_at = time.monotonic() + delay
        detail = (
            str(exc)
            if isinstance(exc, (BotNetworkError, BotAPIError))
            else type(exc).__name__
        )
        health = {
            "state": "retrying",
            "error": detail,
            "attempt": self.poll_failures,
            "retry_seconds": delay,
            "at": self.clock().isoformat(),
        }
        self.journal.set("poll_health", health)
        if not self.transient_read_error(exc):
            self.pause(
                f"Получение обновлений остановлено: {detail}. Проверь настройки; /resume после исправления.",
                reason="poll_error",
            )
        elif (
            self.poll_failures == 1
            or detail != self._last_poll_error
            or time.monotonic() - self._last_poll_log >= 60
        ):
            self.emit(
                "BUSINESS_RECONNECT",
                f"Telegram недоступен ({detail}). Повтор через {delay} сек.; после восстановления продолжим автоматически.",
                **health,
            )
            self._last_poll_log = time.monotonic()
        self._last_poll_error = detail
        return delay

    def poll_succeeded(self) -> None:
        recovered = self.poll_failures > 0
        self.network_available, self.poll_failures, self.poll_retry_at = True, 0, 0.0
        if self.pipeline:
            self.pipeline.delivery_deferred = False
        self.journal.set("poll_health", {"state": "ok", "at": self.clock().isoformat()})
        if recovered:
            self.emit(
                "BUSINESS_RECONNECTED",
                "Связь с Telegram восстановлена. Ручная пауза, если была, сохранена.",
            )

    async def poll(self) -> None:
        updates = await self.api.call(
            "getUpdates",
            offset=self.journal.get("offset", 0),
            timeout=0,
            limit=100,
            allowed_updates=BUSINESS_UPDATES,
        )
        self.journal.queue_updates(updates)
        self.backlog = len(updates) == 100
        await self.process_updates()
        self.poll_succeeded()
        self.journal.write_status()

    async def process_updates(self) -> None:
        if self.journal.unresolved():
            return  # Preserve pre-send state for ACK recovery, keep queued events.
        for update in self.journal.queued():
            await self.handle_update(update)
            self.journal.applied(
                update["update_id"],
                self.pipeline.bird if self.pipeline else None,
                (
                    self.pipeline.paused
                    if self.pipeline
                    else self.journal.get("paused", True)
                ),
            )

    async def handle_update(self, update: dict) -> None:
        if "business_connection" in update:
            connection = update["business_connection"]
            if self.store_connection(connection) and (
                not connection.get("is_enabled")
                or not connection.get("rights", {}).get("can_reply")
            ):
                self.pause(
                    "Подключение отключено или право can_reply отозвано.",
                    reason="connection_rights",
                )
            return
        kind = next((key for key in BUSINESS_UPDATES[1:] if key in update), None)
        if kind is None:
            return
        message = update[kind]
        connection_id = message.get("business_connection_id")
        connection = self.journal.get("connection")
        if not connection or connection["id"] != connection_id:
            try:
                connection = await self.refresh_connection(connection_id)
            except SetupError:
                return
        chat = message.get("chat", {})
        recipient = chat.get("id")
        if chat.get("type") != "private" or recipient not in self.allowed:
            self.emit(
                "BUSINESS_CHAT_IGNORED",
                "Чат не включён в разрешённый список; в модель не передан.",
                chat_id=recipient,
            )
            return
        if kind in ("edited_business_message", "deleted_business_messages"):
            self.journal.set(
                "review_required",
                {
                    "update_id": update["update_id"],
                    "kind": kind,
                    "chat_id": recipient,
                    "message_ids": message.get(
                        "message_ids", [message.get("message_id")]
                    ),
                },
            )
            self.pause(
                "Сообщение изменено/удалено. Автоматика на паузе до сверки истории и памяти; /review — детали.",
                reason="history_review",
            )
            self.emit(
                "BUSINESS_HISTORY_CHANGED",
                "Получено изменение истории; старые факты автоматически не переписываются.",
                kind=kind,
                update_id=update["update_id"],
            )
            return
        sender = message.get("from", {}).get("id")
        bot = self.journal.get("bot", {})
        if message.get("sender_business_bot", {}).get("id") == bot.get("id"):
            return
        if sender == self.owner_id:
            if self.pipeline and any(
                m.message_id == message["message_id"]
                for m in self.pipeline.bird.live.messages
            ):
                self.forget_message(message)
                return
            if self.journal.known_message(message["message_id"]) and not message.get(
                "_persona_test"
            ):
                return
            details = {
                "message_id": message["message_id"],
                "sender_id": sender,
                "message_at": datetime.fromtimestamp(
                    message["date"], timezone.utc
                ).isoformat(),
            }
            if self.pause_on_manual_outgoing:
                self.pause(
                    f"Получено твоё исходящее сообщение {message['message_id']}. /resume — вернуть управление персоне.",
                    reason="owner_message",
                    **details,
                )
            else:
                self.emit(
                    "BUSINESS_MANUAL",
                    "Твоё исходящее добавлено в контекст; автоматическая пауза отключена настройкой.",
                    **details,
                )
            self.remember_message(message)
            if self.pipeline and not any(
                m.message_id == message["message_id"]
                for m in self.pipeline.bird.live.messages
            ):
                bird = self.pipeline.bird
                bird.scheduler.reply_due = None
                bird.scheduler.last_outgoing = self.clock()
                bird.scheduler.awaiting_reply = True
                bird.scheduler.revision += 1
                bird.live.append(
                    Message(
                        bird.history.user,
                        datetime.fromtimestamp(message["date"], timezone.utc),
                        text=message.get("text") or "[вложение]",
                        message_id=message["message_id"],
                    )
                )
                self.pipeline.persist()
            if self.pipeline:
                self.forget_message(message)
            return
        if sender != recipient or message.get("from", {}).get("is_bot"):
            return
        windows = self.journal.get("last_incoming", {})
        key = f"{connection_id}:{recipient}"
        previous = windows.get(key, {})
        if message["date"] > previous.get("date", 0) or (
            message["date"] == previous.get("date", 0)
            and message["message_id"] > previous.get("message_id", 0)
        ):
            windows[key] = {
                "date": message["date"],
                "message_id": message["message_id"],
                "chat_id": recipient,
            }
            self.journal.set("last_incoming", windows)
        if update["update_id"] >= 0:
            self.journal.set(
                "verified_recipient",
                {
                    "chat_id": recipient,
                    "source": "business_message",
                    "update_id": update["update_id"],
                },
            )
        text = message.get("text") or message.get("caption")
        self.remember_message(message)
        if self.pipeline:
            bird = self.pipeline.bird
            if text and not any(
                m.message_id == message["message_id"] for m in bird.live.messages
            ):
                self.pipeline.receive(
                    text,
                    message_id=message["message_id"],
                    timestamp=datetime.fromtimestamp(message["date"], timezone.utc),
                )
            elif not text:
                self.emit(
                    "BUSINESS_ATTACHMENT",
                    "Входящее вложение учтено для окна Telegram; распознавание пока не подключено.",
                    message_id=message["message_id"],
                )
            self.forget_message(message)
            if (
                self.journal.get("auto_pause_reason")
                == "no_verified_incoming_within_24h"
                and self.send_enabled()
                and self.journal.get("appearance_verified", {}).get("connection_id")
                == connection_id
            ):
                self.pipeline.paused = self.pipeline.delivery_blocked = False
                self.journal.set("auto_pause_reason", None)
                self.checkpoint()
                self.emit(
                    "BUSINESS_RESUMED",
                    "Новое входящее открыло окно Telegram; ответ будет создан заново. Старый неотправленный текст не переотправляем.",
                )
        self.emit(
            "BUSINESS_INCOMING",
            "Получено реальное входящее в разрешённом чате.",
            chat_id=recipient,
            message_id=message["message_id"],
        )

    @staticmethod
    def message_key(message: dict) -> str:
        return f"{message['business_connection_id']}:{message['chat']['id']}:{message['message_id']}"

    def remember_message(self, message: dict) -> None:
        pending = self.journal.get("pending_messages", {})
        pending[self.message_key(message)] = message
        self.journal.set("pending_messages", pending)

    def forget_message(self, message: dict) -> None:
        pending = self.journal.get("pending_messages", {})
        pending.pop(self.message_key(message), None)
        with self.journal.db:
            if self.pipeline:
                self.journal._set("state", self.pipeline.bird.runtime_store.snapshot())
            self.journal._set("pending_messages", pending)

    async def restore_messages(self) -> None:
        if self.pipeline and not self.journal.unresolved():
            receipt = self.journal.get("test_receipt")
            if receipt and not any(
                m.message_id == receipt["message_id"]
                for m in self.pipeline.bird.live.messages
            ):
                row = self.journal.row(receipt["request_id"])
                if row:
                    self.remember_message(
                        {
                            "business_connection_id": receipt["connection_id"],
                            "chat": {"id": receipt["recipient_id"], "type": "private"},
                            "from": {"id": self.owner_id},
                            "message_id": receipt["message_id"],
                            "date": int(
                                datetime.fromisoformat(receipt["at"]).timestamp()
                            ),
                            "text": row["text"],
                            "_persona_test": True,
                        }
                    )
            for message in sorted(
                self.journal.get("pending_messages", {}).values(),
                key=lambda row: (row["date"], row["message_id"]),
            ):
                await self.handle_update({"update_id": -1, "business_message": message})

    async def send_message(self, recipient_id: int, text: str) -> bool:
        """Send once through Business Connection; retain every unsent intent."""
        connection = self.journal.get("connection", {})
        window = self.journal.get("last_incoming", {}).get(
            f"{connection.get('id')}:{recipient_id}", {}
        )
        if self._staged:
            identity, payload = self._staged
        else:
            signature = json.dumps(
                [connection.get("id"), recipient_id, text, window.get("message_id")],
                ensure_ascii=False,
            )
            identity = "external-" + hashlib.sha256(signature.encode()).hexdigest()[:24]
            payload = {
                "kind": "external",
                "recipient_id": recipient_id,
                "business_connection_id": connection.get("id"),
            }
        existing = self.journal.row(identity)
        if existing:
            self.last_delivery = {
                "request_id": identity,
                "status": existing["status"],
                "message_id": existing["message_id"],
                "reason": existing["error"],
            }
            return existing["status"] in ("sent", "applied")
        self.journal.prepare(identity, text, payload)

        def block(reason: str, *, retryable: bool = False) -> bool:
            if reason == "no_verified_incoming_within_24h":
                self.journal.set("auto_pause_reason", reason)
            self.journal.update(identity, "blocked", error=reason)
            self.last_delivery = {
                "request_id": identity,
                "status": "blocked",
                "reason": reason,
                "retryable": retryable,
            }
            self.emit(
                "BUSINESS_UNSENT",
                "Сообщение сохранено без отправки: " + reason,
                request_id=identity,
                text=text,
            )
            return False

        if recipient_id not in self.allowed:
            return block("recipient_not_allowlisted")
        if not self.send_enabled():
            return block("PERSONA_SEND_ENABLED=false")
        if not text.strip() or len(text.encode("utf-16-le")) // 2 > 4096:
            return block("invalid_text_length")
        if self.journal.unresolved() and any(
            row["id"] != identity for row in self.journal.unresolved()
        ):
            return block("previous_delivery_unresolved")
        if self.journal.get("review_required"):
            return block("history_review_required")
        if self.journal.get("retry_after", 0) > time.time():
            return block("telegram_retry_after")
        if not connection.get("id"):
            return block("missing_business_connection")
        try:
            connection = await self.refresh_connection(connection["id"])
        except Exception as exc:
            if self.transient_read_error(exc):
                self.poll_failed(exc)
                return block(f"connection_check_failed:{exc}", retryable=True)
            return block(f"connection_check_failed:{type(exc).__name__}")
        if not connection.get("is_enabled"):
            return block("business_connection_disabled")
        if connection.get("rights", {}).get("can_reply") is not True:
            return block("missing_right_can_reply")
        age = self.clock().timestamp() - window.get("date", 0)
        if not window or age < 0 or age >= 86400:
            return block("no_verified_incoming_within_24h")
        # The first explicitly approved test is the only permitted pre-verification send.
        approval = self.journal.get("test_approval", {})
        is_test = (
            identity == approval.get("id")
            and approval.get("approved") is True
            and text == approval.get("text")
            and recipient_id == approval.get("recipient_id")
            and connection["id"] == approval.get("connection_id")
        )
        verified = self.journal.get("appearance_verified") or {}
        if not is_test and (
            verified.get("connection_id") != connection["id"]
            or verified.get("recipient_id") != recipient_id
            or verified.get("sender_id") != self.owner_id
            or verified.get("owner_confirmed") is not True
        ):
            return block("first_test_and_owner_visual_confirmation_required")
        if not is_test and self.journal.get("paused", False):
            return block("persona_paused")
        self.journal.update(identity, "sending")
        try:
            result = await self.api.call(
                "sendMessage",
                business_connection_id=connection["id"],
                chat_id=recipient_id,
                text=text,
            )
        except BotAPIError as exc:
            status = "unknown" if exc.code >= 500 else "rejected"
            reason = f"Bot API {exc.code}: {exc}"
            self.journal.update(identity, status, error=reason)
            if exc.retry_after:
                self.journal.set("retry_after", time.time() + exc.retry_after)
            self.last_delivery = {
                "request_id": identity,
                "status": status,
                "reason": reason,
            }
            self.emit("BUSINESS_UNSENT", reason, request_id=identity, text=text)
            return False
        except Exception as exc:
            self.journal.update(identity, "unknown", error=type(exc).__name__)
            self.last_delivery = {
                "request_id": identity,
                "status": "unknown",
                "reason": type(exc).__name__,
            }
            self.emit(
                "BUSINESS_DELIVERY_UNKNOWN",
                "Результат сети неизвестен. Автоматического повтора нет.",
                request_id=identity,
            )
            return False
        message_id = result.get("message_id")
        if (
            type(message_id) is not int
            or result.get("chat", {}).get("id") != recipient_id
            or result.get("from", {}).get("id") != self.owner_id
            or result.get("business_connection_id") != connection["id"]
        ):
            self.journal.update(
                identity,
                "unknown",
                message_id=message_id if type(message_id) is int else None,
                error="ack_sender_or_connection_mismatch",
            )
            self.last_delivery = {
                "status": "unknown",
                "request_id": identity,
                "reason": "ack_sender_or_connection_mismatch",
            }
            return False
        self.journal.update(identity, "sent", message_id=message_id)
        self.last_delivery = {
            "request_id": identity,
            "status": "sent",
            "message_id": message_id,
        }
        if is_test:
            self.journal.set(
                "test_receipt",
                {
                    "request_id": identity,
                    "message_id": message_id,
                    "recipient_id": recipient_id,
                    "connection_id": connection["id"],
                    "sender_id": self.owner_id,
                    "at": self.clock().isoformat(),
                },
            )
        if "after" not in payload:
            self.journal.update(identity, "applied", message_id=message_id)
        self.emit(
            "BUSINESS_ACK",
            f"Telegram подтвердил сообщение {message_id} от аккаунта владельца.",
            **self.last_delivery,
        )
        return True

    def recover(self) -> None:
        for row in self.journal.unresolved():
            if row["status"] != "sent":
                continue
            payload = json.loads(row["payload"])
            if "after" in payload and self.pipeline:
                self.pipeline.bird.delivery_coordinator.apply_delivery(
                    payload, row["message_id"]
                )
                self.journal.complete(row["id"], self.pipeline.bird)
                self.emit(
                    "BUSINESS_RECOVERED",
                    "Подтверждённая отправка восстановлена без повторного sendMessage.",
                    message_id=row["message_id"],
                )
            elif "after" not in payload:
                if payload.get("kind") == "test" and not self.journal.get(
                    "test_receipt"
                ):
                    self.journal.set(
                        "test_receipt",
                        {
                            "request_id": row["id"],
                            "message_id": row["message_id"],
                            "recipient_id": payload["recipient_id"],
                            "connection_id": payload["connection_id"],
                            "sender_id": self.owner_id,
                            "at": row["created_at"],
                        },
                    )
                self.journal.update(row["id"], "applied")
        if self.journal.unresolved() and self.pipeline:
            self.pipeline.paused = self.pipeline.delivery_blocked = True
            self.emit(
                "BUSINESS_DELIVERY_UNKNOWN",
                "Есть незавершённая отправка. /delivery — состояние; до сверки новых отправок нет.",
            )

    def deliver(
        self, decision: Decision, draft: Draft, now: datetime, path: Path
    ) -> dict | None:
        assert self.pipeline and self.loop
        try:
            try:
                self.loop.run_until_complete(self.poll())
            except Exception as exc:
                if not self.transient_read_error(exc):
                    raise
                self.poll_failed(exc)
                raise DeliveryDeferred(
                    "Проверка входящих временно недоступна; отправка не начата. После восстановления соберём свежий ответ."
                ) from None
            if (
                self.backlog
                or self.pipeline.paused
                or not self.pipeline.bird.scheduler.is_pending(decision)
            ):
                raise DeliverySuperseded()
            prepared = self.pipeline.bird.delivery_coordinator.prepare_delivery(
                decision, draft, self.clock()
            )
            self._staged = (path.stem, prepared)
            if not self.loop.run_until_complete(
                self.send_message(next(iter(self.allowed)), draft.text)
            ):
                if self.last_delivery.get("retryable"):
                    raise DeliveryDeferred(
                        "Не удалось проверить подключение до отправки. Ожидаем восстановления Telegram; сообщение не отправлялось."
                    )
                raise DeliveryBlocked(
                    "Business sendMessage не выполнен: "
                    + self.last_delivery.get("reason", "unknown")
                )
            self.pipeline.bird.delivery_coordinator.apply_delivery(
                prepared, self.last_delivery["message_id"]
            )
            self.journal.complete(path.stem, self.pipeline.bird)
            return prepared["life_change"]
        except (DeliveryBlocked, DeliveryDeferred, DeliverySuperseded):
            raise
        except Exception as exc:
            raise DeliveryBlocked(
                f"Business transport: {type(exc).__name__}. Автоматический повтор остановлен; проверь журнал."
            ) from None
        finally:
            self._staged = None

    def prepare_test(self, recipient_id: int, text: str) -> dict:
        if recipient_id not in self.allowed:
            raise SetupError("Тест разрешён только в явно указанном чате.")
        connection = self.journal.get("connection", {})
        if not connection.get("id"):
            raise SetupError(
                "Сначала получи реальную Business Connection через observe."
            )
        approval = {
            "id": "test-" + os.urandom(12).hex(),
            "recipient_id": recipient_id,
            "text": text,
            "connection_id": connection["id"],
            "owner_id": self.owner_id,
            "approved": False,
        }
        self.journal.set("test_approval", approval)
        return approval
