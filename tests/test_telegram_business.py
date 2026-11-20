"""Official Business API + real pipeline. No external network or real account."""

import asyncio
import json
from copy import deepcopy
from datetime import timedelta
from zipfile import ZipFile

import httpx
import pytest
from test_pipeline import Clock, Model, bird_for, pump

from mockingbird.chat.pipeline import ChatPipeline
from mockingbird.interfaces.telegram.business import (
    BotAPIError,
    BotNetworkError,
    BusinessBotAPI,
    BusinessJournal,
    TelegramBusinessTransport,
)
from mockingbird.interfaces.telegram.state import SetupError
from mockingbird.observability import export_bundle


def connection(**overrides):
    return {
        "id": "test-connection",
        "user": {"id": 10},
        "user_chat_id": 999,
        "date": 1,
        "is_enabled": True,
        "rights": {"can_reply": True},
        **overrides,
    }


class FakeAPI:
    def __init__(self, clock):
        self.clock, self.calls, self.updates = clock, [], []
        self.connection = connection()
        self.error = None
        self.before_send = lambda: None

    async def call(self, method, **params):
        self.calls.append((method, params))
        if method == "getMe":
            return {
                "id": 777,
                "is_bot": True,
                "username": "mockingbird_persona_bot",
                "can_connect_to_business": True,
            }
        if method == "getWebhookInfo":
            return {"url": ""}
        if method == "getBusinessConnection":
            return deepcopy(self.connection)
        if method == "getUpdates":
            return [
                row for row in self.updates if row["update_id"] >= params["offset"]
            ][:100]
        if method == "sendMessage":
            self.before_send()
            if self.error:
                raise self.error
            return {
                "message_id": 200,
                "chat": {"id": params["chat_id"], "type": "private"},
                "from": {"id": 10},
                "business_connection_id": params["business_connection_id"],
                "text": params["text"],
            }
        raise AssertionError(method)

    def incoming(
        self, identity=1, text="привет", sender=20, chat=20, kind="business_message"
    ):
        self.updates.append(
            {
                "update_id": identity,
                kind: {
                    "business_connection_id": "test-connection",
                    "message_id": identity,
                    "date": int(self.clock().timestamp()),
                    "chat": {"id": chat, "type": "private"},
                    "from": {"id": sender},
                    "text": text,
                },
            }
        )


@pytest.fixture
def business(tmp_path, monkeypatch):
    monkeypatch.setenv("PERSONA_SEND_ENABLED", "true")
    monkeypatch.delenv("TELEGRAM_BUSINESS_CONNECTION_ID", raising=False)
    clock, loop = Clock(), asyncio.new_event_loop()
    api = FakeAPI(clock)
    journal = BusinessJournal(tmp_path, (10, 20))
    transport = TelegramBusinessTransport(
        api, journal, owner_id=10, allowed_chat_ids=[20], clock=clock
    )
    transport.loop = loop
    loop.run_until_complete(transport.initialize())
    journal.set("connection", connection())
    journal.set("paused", False)
    journal.set(
        "appearance_verified",
        {
            "connection_id": "test-connection",
            "recipient_id": 20,
            "sender_id": 10,
            "owner_confirmed": True,
        },
    )
    yield transport, clock, api, loop
    if transport.pipeline:
        transport.pipeline.close(checkpoint_life=not bool(journal.unresolved()))
    journal.close()
    loop.close()


def attach(transport, clock):
    events = []
    pipeline = ChatPipeline(
        bird_for(Model()),
        transport.journal.directory,
        events.append,
        clock=clock,
        delivery=transport.deliver,
        state_sink=transport.checkpoint,
    )
    transport.pipeline = pipeline
    pipeline.start()
    return pipeline, events


def incoming(business):
    transport, clock, api, loop = business
    api.incoming()
    loop.run_until_complete(transport.poll())


def test_connection_owner_rights_and_user_chat_id_separation(business):
    transport, clock, api, loop = business
    incoming(business)
    assert loop.run_until_complete(transport.send_message(20, "ответ"))
    payload = next(params for method, params in api.calls if method == "sendMessage")
    assert payload == {
        "business_connection_id": "test-connection",
        "chat_id": 20,
        "text": "ответ",
    }
    assert payload["chat_id"] != transport.journal.get("connection")["user_chat_id"]


@pytest.mark.parametrize(
    "restriction",
    [
        "disabled",
        "rights",
        "expired",
        "no-incoming",
        "switch",
        "not-allowed",
        "first-test",
        "wrong-owner",
    ],
)
def test_preflight_never_sends_on_restrictions(business, monkeypatch, restriction):
    transport, clock, api, loop = business
    if restriction != "no-incoming":
        incoming(business)
    recipient = 20
    if restriction == "disabled":
        api.connection["is_enabled"] = False
    if restriction == "rights":
        api.connection["rights"] = {}
    if restriction == "expired":
        clock.now += timedelta(hours=24)
    if restriction == "switch":
        monkeypatch.setenv("PERSONA_SEND_ENABLED", "false")
    if restriction == "not-allowed":
        recipient = 999
    if restriction == "first-test":
        transport.journal.set("appearance_verified", {})
    if restriction == "wrong-owner":
        api.connection["user"]["id"] = 12345
    assert not loop.run_until_complete(
        transport.send_message(recipient, "неотправленное")
    )
    assert not any(method == "sendMessage" for method, _ in api.calls)
    assert tuple(
        transport.journal.db.execute("SELECT status,text FROM outbox").fetchone()
    ) == ("blocked", "неотправленное")


def test_incoming_window_is_not_renewed_by_owner_or_edit(business):
    transport, clock, api, loop = business
    incoming(business)
    original = transport.journal.get("last_incoming")
    clock.now += timedelta(hours=25)
    api.incoming(2, sender=10)
    api.incoming(3, kind="edited_business_message")
    loop.run_until_complete(transport.poll())
    assert transport.journal.get("last_incoming") == original
    assert transport.journal.get("review_required")["kind"] == "edited_business_message"


def test_deleted_messages_pause_and_duplicate_update_is_applied_once(business):
    transport, clock, api, loop = business
    pipeline, events = attach(transport, clock)
    incoming(business)
    revision = pipeline.bird.scheduler.revision
    transport.journal.queue_updates(api.updates)
    loop.run_until_complete(transport.process_updates())
    assert len(pipeline.bird.live.messages) == 1
    assert pipeline.bird.scheduler.revision == revision
    api.updates.append(
        {
            "update_id": 2,
            "deleted_business_messages": {
                "business_connection_id": "test-connection",
                "chat": {"id": 20, "type": "private"},
                "message_ids": [1],
            },
        }
    )
    loop.run_until_complete(transport.poll())
    assert pipeline.paused
    assert transport.journal.get("review_required")["message_ids"] == [1]


def test_observe_keeps_incoming_for_later_persona(business):
    transport, clock, api, loop = business
    incoming(business)
    assert transport.journal.get("pending_messages")
    pipeline, events = attach(transport, clock)
    loop.run_until_complete(transport.restore_messages())
    loop.run_until_complete(transport.restore_messages())
    assert len(pipeline.bird.live.messages) == 1
    assert not transport.journal.get("pending_messages")
    assert transport.journal.get("verified_recipient")["update_id"] == 1


def test_pipeline_ack_precedes_history_commit(business):
    transport, clock, api, loop = business
    pipeline, events = attach(transport, clock)
    incoming(business)

    def uncommitted():
        assert len(pipeline.bird.live.messages) == 1
        assert pipeline.bird.scheduler.reply_due is not None

    api.before_send = uncommitted
    clock.now += timedelta(seconds=5)
    pipeline.step()
    pump(pipeline, lambda: any(e.stage == "OUTPUT" for e in events))
    assert pipeline.bird.live.messages[-1].message_id == 200
    assert pipeline.bird.scheduler.reply_due is None
    assert transport.journal.get("state") == pipeline.bird.runtime_store.snapshot()


def test_timeout_stops_retries_and_persists_unknown(business):
    transport, clock, api, loop = business
    pipeline, events = attach(transport, clock)
    incoming(business)
    api.error = BotNetworkError("ReadTimeout")
    clock.now += timedelta(seconds=5)
    pipeline.step()
    pump(pipeline, lambda: pipeline.delivery_blocked)
    assert transport.journal.unresolved()[0]["status"] == "unknown"
    assert len(pipeline.bird.live.messages) == 1
    for _ in range(5):
        pipeline.step()
    assert len([call for call in api.calls if call[0] == "sendMessage"]) == 1


def test_public_send_deduplicates_across_restart(business):
    transport, clock, api, loop = business
    incoming(business)
    assert loop.run_until_complete(transport.send_message(20, "один раз"))
    second = TelegramBusinessTransport(
        api, transport.journal, owner_id=10, allowed_chat_ids=[20], clock=clock
    )
    assert loop.run_until_complete(second.send_message(20, "один раз"))
    assert len([call for call in api.calls if call[0] == "sendMessage"]) == 1


def test_confirmed_send_recovers_without_new_send(business, monkeypatch):
    transport, clock, api, loop = business
    pipeline, events = attach(transport, clock)
    incoming(business)
    apply = pipeline.bird.delivery_coordinator.apply_delivery
    monkeypatch.setattr(
        pipeline.bird.delivery_coordinator,
        "apply_delivery",
        lambda *args: (_ for _ in ()).throw(OSError("disk")),
    )
    clock.now += timedelta(seconds=5)
    pipeline.step()
    pump(pipeline, lambda: pipeline.delivery_blocked)
    assert transport.journal.unresolved()[0]["status"] == "sent"
    monkeypatch.setattr(pipeline.bird.delivery_coordinator, "apply_delivery", apply)
    transport.recover()
    transport.recover()
    assert not transport.journal.unresolved()
    assert len(pipeline.bird.live.messages) == 2
    assert len([call for call in api.calls if call[0] == "sendMessage"]) == 1


@pytest.mark.parametrize("code", [400, 403, 429, 500])
def test_rejection_is_not_retried(business, code):
    transport, clock, api, loop = business
    incoming(business)
    api.error = BotAPIError(code, "reply not allowed", 120 if code == 429 else 0)
    assert not loop.run_until_complete(transport.send_message(20, "текст"))
    assert not loop.run_until_complete(transport.send_message(20, "текст"))
    assert len([call for call in api.calls if call[0] == "sendMessage"]) == 1


def test_first_test_needs_explicit_exact_approval(business):
    transport, clock, api, loop = business
    incoming(business)
    transport.journal.set("appearance_verified", {})
    approval = transport.prepare_test(20, "Привет, Вань!")
    assert approval["approved"] is False
    assert not any(method == "sendMessage" for method, _ in api.calls)
    approval["approved"] = True
    transport.journal.set("test_approval", approval)
    transport._staged = (approval["id"], {"kind": "test"})
    assert loop.run_until_complete(transport.send_message(20, approval["text"]))
    assert transport.journal.get("test_receipt")["sender_id"] == 10
    assert not transport.journal.get("appearance_verified")


def test_approved_test_is_restored_into_conversation_without_resending(business):
    transport, clock, api, loop = business
    incoming(business)
    approval = transport.prepare_test(20, "проверка подключения")
    approval["approved"] = True
    transport.journal.set("test_approval", approval)
    transport._staged = (approval["id"], {"kind": "test", **approval})
    assert loop.run_until_complete(transport.send_message(20, approval["text"]))
    pipeline, events = attach(transport, clock)
    loop.run_until_complete(transport.restore_messages())
    loop.run_until_complete(transport.restore_messages())
    assert [m.author for m in pipeline.bird.live.messages] == ["Partner", "Persona"]
    assert pipeline.bird.live.messages[-1].text == approval["text"]
    assert pipeline.bird.scheduler.reply_due is None
    assert len([call for call in api.calls if call[0] == "sendMessage"]) == 1


def test_http_errors_never_expose_token(monkeypatch):
    token = "12345:PRIVATE_TEST_TOKEN"
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", token)

    def handler(request):
        assert request.url.host == "api.telegram.org"
        return httpx.Response(
            400, json={"ok": False, "error_code": 400, "description": "bad " + token}
        )

    async def check():
        api = BusinessBotAPI(token, transport=httpx.MockTransport(handler))
        try:
            with pytest.raises(BotAPIError) as caught:
                await api.call("getMe")
            assert token not in str(caught.value)
        finally:
            await api.close()

    asyncio.run(check())


def test_mass_messaging_is_rejected(business):
    transport, clock, api, loop = business
    with pytest.raises(SetupError):
        TelegramBusinessTransport(
            api, transport.journal, owner_id=10, allowed_chat_ids=[20, 30]
        )


def test_new_incoming_reopens_only_automatic_window_pause(business):
    transport, clock, api, loop = business
    pipeline, events = attach(transport, clock)
    incoming(business)
    clock.now += timedelta(hours=24)
    assert not loop.run_until_complete(transport.send_message(20, "старая инициатива"))
    pipeline.paused = pipeline.delivery_blocked = True
    api.incoming(2, text="новый вопрос")
    loop.run_until_complete(transport.poll())
    assert not pipeline.paused
    assert not pipeline.delivery_blocked
    assert not any(method == "sendMessage" for method, _ in api.calls)
    transport.pause("Ручная пауза")
    api.incoming(3)
    loop.run_until_complete(transport.poll())
    assert pipeline.paused


def test_atomic_snapshot_restores_despite_incomplete_legacy_exports(tmp_path):
    bird = bird_for(Model())
    bird.receive("вопрос", Clock()())
    bird.runtime_store.save(tmp_path)
    (tmp_path / "scheduler.json").unlink()
    restored = bird_for(Model())
    restored.runtime_store.restore(tmp_path)
    assert restored.runtime_store.snapshot() == bird.runtime_store.snapshot()


def test_export_keeps_diagnostics_without_credentials(business, monkeypatch):
    transport, clock, api, loop = business
    root = transport.journal.directory
    token = "12345:PRIVATE_BUSINESS_TOKEN"
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", token)
    (root / ".env").write_text("TELEGRAM_BOT_TOKEN=" + token)
    (root / "private.session").write_text("auth session")
    (root / "decisions.jsonl").write_text(
        json.dumps({"stage": "ERROR", "message": token}) + "\n"
    )
    transport.journal.report()
    path = export_bundle(root)
    with ZipFile(path) as archive:
        assert "runtime/business_status.json" in archive.namelist()
        assert not any(
            ".env" in name or ".session" in name or "sqlite" in name
            for name in archive.namelist()
        )
        assert all(
            token not in archive.read(name).decode() for name in archive.namelist()
        )
