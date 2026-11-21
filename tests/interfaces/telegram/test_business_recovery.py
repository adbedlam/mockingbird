"""Read-side recovery must never retry an uncertain send or release manual pause."""

import json
from datetime import timedelta

import httpx
import pytest
from tests.chat.test_pipeline import pump
from tests.interfaces.telegram.test_telegram_business import attach, business, incoming

from mockingbird.interfaces.telegram.api import (
    BotAPIError,
    BotNetworkError,
    TelegramBotAPI,
)


def test_poll_outage_does_not_persist_manual_pause_or_generate(business):
    transport, clock, api, loop = business
    pipeline, events = attach(transport, clock)
    incoming(business)
    clock.now += timedelta(seconds=5)
    delay = transport.poll_failed(
        BotNetworkError("ConnectTimeout", method="getUpdates")
    )
    assert delay == 15
    assert not pipeline.paused
    assert not transport.journal.get("paused")
    pipeline.step()
    assert not pipeline.mockingbird.model.calls
    assert not any(e.stage == "PAUSED" for e in events)
    loop.run_until_complete(transport.poll())
    assert transport.network_available
    assert not pipeline.paused
    pipeline.step()
    pump(pipeline, lambda: any(e.stage == "OUTPUT" for e in events))
    assert len([call for call in api.calls if call[0] == "sendMessage"]) == 1


def test_network_recovery_preserves_manual_owner_pause(business):
    transport, clock, api, loop = business
    pipeline, events = attach(transport, clock)
    api.incoming(sender=10)
    loop.run_until_complete(transport.poll())
    assert pipeline.paused
    assert transport.journal.get("pause_reason") == "owner_message"
    transport.poll_failed(BotNetworkError("ReadTimeout", method="getUpdates"))
    loop.run_until_complete(transport.poll())
    assert pipeline.paused
    assert transport.journal.get("pause_reason") == "owner_message"
    assert not pipeline.mockingbird.model.calls


def test_reconnect_backoff_throttles_duplicate_logs_and_honors_429(business):
    transport, clock, api, loop = business
    pipeline, events = attach(transport, clock)
    delays = [
        transport.poll_failed(BotNetworkError("ConnectTimeout")) for _ in range(8)
    ]
    assert delays == [15, 30, 60, 60, 60, 60, 60, 60]
    assert len([e for e in events if e.stage == "BUSINESS_RECONNECT"]) == 1
    assert not any(e.stage == "PAUSED" for e in events)
    assert transport.poll_failed(BotAPIError(429, "retry", 120)) == 120


def test_auth_failure_requires_operator_attention(business):
    transport, clock, api, loop = business
    pipeline, events = attach(transport, clock)
    transport.poll_failed(BotAPIError(401, "Unauthorized"))
    assert pipeline.paused
    assert transport.journal.get("pause_reason") == "poll_error"
    loop.run_until_complete(transport.poll())
    assert pipeline.paused


def test_repeated_owner_update_does_not_pause_after_resume(business):
    transport, clock, api, loop = business
    pipeline, events = attach(transport, clock)
    api.incoming(sender=10)
    loop.run_until_complete(transport.poll())
    pipeline.command("/resume")
    transport.checkpoint()
    count = len([e for e in events if e.stage == "BUSINESS_PAUSED"])
    # A different update_id can still carry an already-recorded message_id.
    duplicate = json.loads(json.dumps(api.updates[0]))
    duplicate["update_id"] = 2
    api.updates.append(duplicate)
    loop.run_until_complete(transport.poll())
    assert not pipeline.paused
    assert len(pipeline.mockingbird.live.messages) == 1
    assert len([e for e in events if e.stage == "BUSINESS_PAUSED"]) == count


def test_manual_outgoing_can_update_context_without_auto_pause(business):
    transport, clock, api, loop = business
    transport.pause_on_manual_outgoing = False
    pipeline, events = attach(transport, clock)
    incoming(business)
    api.incoming(2, text="я сам ответил", sender=10)
    loop.run_until_complete(transport.poll())
    assert not pipeline.paused
    assert pipeline.mockingbird.scheduler.reply_due is None
    assert pipeline.mockingbird.live.messages[-1].text == "я сам ответил"
    api.incoming(3, text="следующий вопрос")
    loop.run_until_complete(transport.poll())
    clock.now += timedelta(seconds=5)
    pipeline.step()
    pump(pipeline, lambda: any(e.stage == "OUTPUT" for e in events))
    assert len([call for call in api.calls if call[0] == "sendMessage"]) == 1


@pytest.mark.parametrize("method", ["getUpdates", "getBusinessConnection"])
def test_pre_send_read_failure_defers_without_permanent_pause(business, method):
    transport, clock, api, loop = business
    pipeline, events = attach(transport, clock)
    incoming(business)
    original = api.call

    async def fail_read(name, **params):
        if name == method:
            raise BotNetworkError("ConnectTimeout", method=name)
        return await original(name, **params)

    api.call = fail_read
    clock.now += timedelta(seconds=5)
    pipeline.step()
    pump(pipeline, lambda: pipeline.delivery_deferred)
    assert not pipeline.paused
    assert not pipeline.delivery_blocked
    assert not transport.journal.unresolved()
    assert not any(name == "sendMessage" for name, _ in api.calls)
    assert len(pipeline.mockingbird.model.calls) == 1
    api.call = original
    loop.run_until_complete(transport.poll())
    pipeline.step()
    pump(pipeline, lambda: any(e.stage == "OUTPUT" for e in events))
    assert len([call for call in api.calls if call[0] == "sendMessage"]) == 1


def test_unknown_send_timeout_is_never_released_by_read_recovery(business):
    transport, clock, api, loop = business
    pipeline, events = attach(transport, clock)
    incoming(business)
    api.error = BotNetworkError("ReadTimeout", method="sendMessage")
    clock.now += timedelta(seconds=5)
    pipeline.step()
    pump(pipeline, lambda: pipeline.delivery_blocked)
    assert transport.journal.unresolved()[0]["status"] == "unknown"
    transport.poll_failed(BotNetworkError("ConnectTimeout", method="getUpdates"))
    loop.run_until_complete(transport.poll())
    assert pipeline.paused and pipeline.delivery_blocked
    assert transport.journal.unresolved()[0]["status"] == "unknown"
    assert len([call for call in api.calls if call[0] == "sendMessage"]) == 1


def test_network_log_includes_safe_method_and_error_without_token(monkeypatch):
    token = "12345:PRIVATE_TEST_TOKEN"
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", token)

    def fail(request):
        raise httpx.ConnectTimeout(f"private URL {request.url}", request=request)

    async def check():
        api = TelegramBotAPI(token, transport=httpx.MockTransport(fail))
        try:
            with pytest.raises(BotNetworkError) as caught:
                await api.call("getUpdates")
            assert str(caught.value) == "getUpdates: ConnectTimeout"
            assert token not in str(caught.value)
        finally:
            await api.close()

    import asyncio

    asyncio.run(check())
