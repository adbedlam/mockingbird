import json
import os
import threading
import time

import httpx
import pytest

from mockingbird.dashboard.app import (
    DashboardData,
    create_server,
    dashboard_key,
    running_dashboard,
    tail_events,
)


@pytest.fixture
def dashboard(tmp_path):
    runtime = tmp_path / "runtime"
    directory = runtime / "business" / "pair123"
    directory.mkdir(parents=True)
    memory, processed = tmp_path / "memory", tmp_path / "processed"
    memory.mkdir()
    processed.mkdir()
    event = {
        "at": "2026-11-18T13:00:00+03:00",
        "stage": "OUTPUT",
        "message": "готово",
        "data": {"text": "<script>bad()</script>", "reason": "отвечаю на вопрос"},
    }
    (directory / "decisions.jsonl").write_text(
        json.dumps(event, ensure_ascii=False) + "\n{in-progress", encoding="utf-8"
    )
    state = {
        "history": {
            "messages": [
                {"author": "Persona", "text": "привет", "timestamp": event["at"]}
            ]
        },
        "scheduler": {"initiations_sent": 1},
    }
    (directory / "runtime_state.json").write_text(json.dumps(state), encoding="utf-8")
    requests = directory / "requests"
    requests.mkdir()
    (requests / ("a" * 32 + ".json")).write_text(
        json.dumps(
            {"id": "a" * 32, "messages": [{"role": "user", "content": "контекст"}]},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return (
        DashboardData(runtime, memory, processed, "pair123", "Persona", "Partner"),
        directory,
    )


def test_snapshot_contains_real_events_and_bounded_history(dashboard):
    data, directory = dashboard
    snapshot = data.snapshot(None)
    assert snapshot["history"][0]["text"] == "привет"
    assert len(snapshot["events"]) == 1
    assert snapshot["events"][0]["data"]["reason"] == "отвечаю на вопрос"
    assert snapshot["scheduler"]["initiations_sent"] == 1
    assert (
        data.request(snapshot["session_id"], "a" * 32)["messages"][0]["content"]
        == "контекст"
    )
    with pytest.raises(ValueError):
        data.request(snapshot["session_id"], "../../.env")
    with pytest.raises(ValueError):
        data.snapshot("../../.env")


def test_legacy_conversation_is_selected_before_empty_connection(dashboard):
    data, directory = dashboard
    legacy = data.runtime / "fast-replies" / "pair123"
    legacy.mkdir(parents=True)
    history = legacy / "chat_history.json"
    history.write_text(
        json.dumps({"messages": [{"author": "Persona", "text": "legacy conversation"}]})
    )
    os.utime(history, (time.time() + 10, time.time() + 10))
    snapshot = data.snapshot(None)
    assert snapshot["history"][0]["text"] == "legacy conversation"


def test_server_requires_key_local_host_and_never_exposes_secret(
    dashboard, monkeypatch
):
    data, directory = dashboard
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "12345:SECRET_TOKEN")
    (directory / "chat_history.json").write_text('{"messages":[]}', encoding="utf-8")
    (directory / "decisions.jsonl").write_text(
        json.dumps({"stage": "ERROR", "message": "12345:SECRET_TOKEN"}) + "\n",
        encoding="utf-8",
    )
    server = create_server(data, "local-test-key", 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        with httpx.Client(trust_env=False) as client:
            assert client.get(base + "/api/state").status_code == 403
            response = client.get(
                base + "/api/state", headers={"X-Dashboard-Key": "local-test-key"}
            )
            assert response.status_code == 200
            assert "SECRET_TOKEN" not in response.text
            assert "[REDACTED]" in response.text
            assert (
                client.get(
                    base + "/api/state",
                    headers={
                        "X-Dashboard-Key": "local-test-key",
                        "Host": "evil.example",
                    },
                ).status_code
                == 403
            )
            page = client.get(base + "/?key=local-test-key")
            assert page.status_code == 200
            assert "Решения модели" in page.text
            assert "textContent" in page.text
            assert "unsafe-eval" not in page.headers["Content-Security-Policy"]
            assert (
                client.post(
                    base + "/api/send", headers={"X-Dashboard-Key": "local-test-key"}
                ).status_code
                == 501
            )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(2)


def test_dashboard_key_stays_valid_between_restarts(tmp_path):
    path = tmp_path / "dashboard" / "access.key"
    first = dashboard_key(path)
    assert len(first) >= 24
    assert dashboard_key(path) == first


def test_dashboard_health_confirms_own_service_and_identity(dashboard):
    data, directory = dashboard
    server = create_server(data, "stable-access-key-for-tests", 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        assert running_dashboard(
            server.server_port, "stable-access-key-for-tests", "pair123"
        )
        assert not running_dashboard(server.server_port, "wrong-key", "pair123")
        assert not running_dashboard(
            server.server_port, "stable-access-key-for-tests", "another-pair"
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(2)
