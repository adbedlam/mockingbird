import json

import pytest
from dotenv import dotenv_values
from test_telegram_business import business, connection

from mockingbird.config import save_business_connection_id
from mockingbird.interfaces.telegram.business import BotAPIError


@pytest.mark.parametrize("newline", ["\n", "\r\n"])
def test_env_update_preserves_other_keys_and_is_idempotent(tmp_path, newline):
    path = tmp_path / ".env"
    original = newline.join(
        [
            "# пользовательские настройки",
            "TELEGRAM_BOT_TOKEN=private-test-value",
            "TELEGRAM_BUSINESS_CONNECTION_ID=",
            "PERSONA_SEND_ENABLED=false",
            "",
        ]
    )
    path.write_bytes(original.encode())
    assert save_business_connection_id("connection_from_api", path)
    values = dotenv_values(path)
    assert values["TELEGRAM_BUSINESS_CONNECTION_ID"] == "connection_from_api"
    assert values["TELEGRAM_BOT_TOKEN"] == "private-test-value"
    assert values["PERSONA_SEND_ENABLED"] == "false"
    assert (
        path.read_bytes()
        .decode()
        .replace(
            "TELEGRAM_BUSINESS_CONNECTION_ID='connection_from_api'",
            "TELEGRAM_BUSINESS_CONNECTION_ID=",
        )
        == original
    )
    before = path.stat().st_mtime_ns
    assert not save_business_connection_id("connection_from_api", path)
    assert path.stat().st_mtime_ns == before


def test_connection_event_persists_only_verified_owner_and_handles_reconnect(
    business, tmp_path
):
    transport, clock, api, loop = business
    transport.env_path = tmp_path / ".env"
    transport.env_path.write_text("TELEGRAM_BOT_TOKEN=preserved\n")
    assert not transport.store_connection(connection(user={"id": 12345}))
    assert "TELEGRAM_BUSINESS_CONNECTION_ID" not in dotenv_values(transport.env_path)
    newer = connection(id="new-connection", date=2)
    api.updates.append({"update_id": 10, "business_connection": newer})
    loop.run_until_complete(transport.poll())
    assert (
        dotenv_values(transport.env_path)["TELEGRAM_BUSINESS_CONNECTION_ID"]
        == "new-connection"
    )
    assert not transport.store_connection(connection(is_enabled=False))
    assert (
        dotenv_values(transport.env_path)["TELEGRAM_BUSINESS_CONNECTION_ID"]
        == "new-connection"
    )
    assert not any(method == "sendMessage" for method, _ in api.calls)


def test_message_can_discover_connection_when_connection_event_is_missing(
    business, tmp_path
):
    transport, clock, api, loop = business
    transport.env_path = tmp_path / ".env"
    transport.journal.set("connection", None)
    api.incoming()
    loop.run_until_complete(transport.poll())
    assert (
        dotenv_values(transport.env_path)["TELEGRAM_BUSINESS_CONNECTION_ID"]
        == "test-connection"
    )
    assert transport.journal.get("verified_recipient")["chat_id"] == 20


def test_stale_bootstrap_id_does_not_stop_discovery(business, tmp_path, monkeypatch):
    transport, clock, api, loop = business
    transport.env_path = tmp_path / ".env"
    monkeypatch.setenv("TELEGRAM_BUSINESS_CONNECTION_ID", "expired-connection")
    original_call = api.call

    async def call(method, **params):
        if (
            method == "getBusinessConnection"
            and params["business_connection_id"] == "expired-connection"
        ):
            raise BotAPIError(400, "BUSINESS_CONNECTION_NOT_FOUND")
        return await original_call(method, **params)

    api.call = call
    loop.run_until_complete(transport.initialize())
    api.updates.append({"update_id": 10, "business_connection": connection()})
    loop.run_until_complete(transport.poll())
    assert (
        dotenv_values(transport.env_path)["TELEGRAM_BUSINESS_CONNECTION_ID"]
        == "test-connection"
    )
