import json
import os
import subprocess
import sys
import threading
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest
import yaml
from test_life import plan
from test_local_persona import config_for

from mockingbird.persona.life import PersonaLife
from mockingbird.persona.refresh import bootstrap_local_persona


@pytest.mark.parametrize("generate", [False, True])
def test_isolated_change_preview_cli_prepare_or_two_local_http_replies(
    tmp_path, generate
):
    config = config_for(tmp_path)
    bootstrap_local_persona(config)
    payloads = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            payloads.append(body)
            if len(payloads) == 1:
                messages = body["messages"]
                protocol = next(
                    m["content"]
                    for m in messages
                    if m["content"].startswith("ИЗМЕНЕНИЯ ПЛАНА")
                )
                incoming = json.loads(protocol.rsplit("\n", 1)[1])
                reply = {
                    "send": True,
                    "text": "давай вечером потестим проект",
                    "reason": "соглашаюсь",
                    "life_update": {
                        "date": "2026-11-17",
                        "base_revision": 0,
                        "source_id": incoming[-1]["id"],
                        "reason": "заменили прогулку проектом",
                        "changes": [
                            {
                                "op": "replace",
                                "id": "a3",
                                "start": "18:00",
                                "end": "19:00",
                                "activity": "потестировал проект",
                            }
                        ],
                    },
                }
            else:
                reply = {
                    "send": True,
                    "text": "вечером проект потестим",
                    "reason": "помню договорённость",
                    "life_update": None,
                }
            response = json.dumps(
                {
                    "id": "test",
                    "object": "chat.completion",
                    "created": 0,
                    "model": "test",
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": json.dumps(reply),
                            },
                            "finish_reason": "stop",
                        }
                    ],
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response)))
            self.end_headers()
            self.wfile.write(response)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    config["inference_model"].update(
        host=f"http://127.0.0.1:{server.server_port}/v1",
        model="test",
        timeout_seconds=5,
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    life = PersonaLife(tmp_path / "original", "Nikita", "Vanya", "Europe/Moscow")
    life.apply(
        plan(),
        "2026-11-17",
        datetime(2026, 11, 17, 13, tzinfo=ZoneInfo("Europe/Moscow")),
    )
    source = life.directory / "life.json"
    before = source.read_bytes()
    output = tmp_path / "preview"
    try:
        result = subprocess.run(
            [
                sys.executable,
                "scripts/preview_life_changes.py",
                "--config",
                str(config_path),
                "--plan",
                str(source),
                "--at",
                "2026-11-17T13:00:00+03:00",
                "--output",
                str(output),
                "--message",
                "давай вместо прогулки с 18 до 19 потестим проект",
                "--generate" if generate else "--prepare",
            ],
            cwd=Path(__file__).resolve().parents[1],
            env={
                **os.environ,
                "PYTHONIOENCODING": "utf-8",
                "INFERENCE_MODEL_API_KEY": "test-only",
                "NO_PROXY": "127.0.0.1,localhost",
            },
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=20,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert source.read_bytes() == before
        if generate:
            assert len(payloads) == 2
            assert "LIFE_CHANGED" in result.stdout
            assert (output / "report.md").exists()
            updated = PersonaLife(output / "life", "Nikita", "Vanya", "Europe/Moscow")
            assert updated.days["2026-11-17"]["revision"] == 1
            assert "потестировал проект" in updated.diary_view()
            assert "погулял" not in updated.diary_view()
            state = next(
                m["content"]
                for m in payloads[1]["messages"]
                if m["content"].startswith("РАЗРЕШЁННАЯ")
            )
            assert '"revision": 1' in state
        else:
            assert payloads == []
            request = json.loads(
                (output / "change.request.json").read_text(encoding="utf-8")
            )
            assert request["body"]["response_format"] == {"type": "json_object"}
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=2)
