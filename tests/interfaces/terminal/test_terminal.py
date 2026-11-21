"""Run the actual CLI against a local OpenAI-compatible fixture, without an API key."""

import json
import os
import subprocess
import sys
import threading
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
import yaml
from tests.config_helpers import scheduler_settings

from mockingbird.conversation.history import History
from mockingbird.conversation.message import Message
from mockingbird.persona.diary.diary import Diary
from mockingbird.persona.inner.inner import InnerPrompt


@pytest.mark.parametrize("with_life", [False, True])
def test_real_terminal_input_status_context_and_unicode_output(
    tmp_path: Path, with_life: bool
):
    received = threading.Event()
    release = threading.Event()
    payloads: list[dict] = []
    reply = "Всё норм, ты как? \U0001f426"

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            payloads.append(
                json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            )
            payload = payloads[-1]
            if payload["messages"][0]["content"].startswith(
                "You are planning the FICTIONAL"
            ):
                day = json.loads(payload["messages"][1]["content"])["date"]
                raw = json.dumps(
                    {
                        "date": day,
                        "activities": [
                            {
                                "start": "00:00",
                                "end": "00:01",
                                "activity": "разобрал файлы",
                            }
                        ],
                        "reason": "Локальный тест",
                    }
                )
            else:
                received.set()
                release.wait(5)
                raw = json.dumps(
                    {"send": True, "text": reply, "reason": "Отвечаю на приветствие"},
                    ensure_ascii=False,
                )
            body = json.dumps(
                {
                    "id": "test",
                    "object": "chat.completion",
                    "created": 0,
                    "model": "local-test",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": raw},
                            "finish_reason": "stop",
                        }
                    ],
                },
                ensure_ascii=False,
            ).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    processed = tmp_path / "processed"
    History(
        "Persona",
        "Partner",
        [Message("Persona", datetime(2026, 10, 1, 9), text="archive")],
    ).save(processed / "history.json")
    InnerPrompt("Synthetic persona").save(processed / "inner.md")
    Diary().save(processed / "diary.json")
    config = {
        "user": "Persona",
        "other": "Partner",
        "inference_model": {
            "host": f"http://127.0.0.1:{server.server_port}/v1",
            "model": "local-test",
            "timeout_seconds": 5,
        },
        "scheduler": scheduler_settings(
            max_initiations_per_day=0,
            min_reply_delay_seconds=1,
        ),
        "life": {"enabled": with_life},
        "paths": {
            "raw": str(tmp_path / "raw"),
            "processed": str(processed),
            "history_filename": "history.json",
            "inner_prompt_filename": "inner.md",
            "diary_filename": "diary.json",
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    env = {
        **os.environ,
        "PYTHONIOENCODING": "utf-8",
        "INFERENCE_MODEL_API_KEY": "test-only",
        "NO_PROXY": "127.0.0.1,localhost",
    }
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "mockingbird.mockingbird",
            "--config",
            str(config_path),
            "--state-dir",
            str(tmp_path / "runtime"),
            "--fast-replies",
            "--max-generations",
            "2" if with_life else "1",
        ],
        cwd=Path(__file__).resolve().parents[3],
        env=env,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
    )
    lines: list[str] = []

    def read_output():
        assert process.stdout is not None
        for line in process.stdout:
            lines.append(line)

    reader = threading.Thread(target=read_output, daemon=True)
    reader.start()

    def wait_for(text: str):
        deadline = time.monotonic() + 12
        while (
            text not in "".join(lines)
            and time.monotonic() < deadline
            and process.poll() is None
        ):
            time.sleep(0.02)
        assert text in "".join(lines), "".join(lines)

    def send(text: str):
        assert process.stdin is not None
        process.stdin.write(text + "\n")
        process.stdin.flush()

    try:
        wait_for("/help")
        send("Привет, как дела? \U0001f426")
        wait_for("SCHEDULE")
        assert received.wait(5), "".join(lines)
        send("/status")
        wait_for("Model request: in progress")
        send("/context")
        wait_for("CONTEXT")
        release.set()
        wait_for(reply)
        send("/quit")
        assert process.wait(timeout=5) == 0
        reader.join(timeout=2)
        assert "STOP" in "".join(lines)
        assert len(payloads) == (2 if with_life else 1)
        assert all(p["response_format"] == {"type": "json_object"} for p in payloads)
        assert "Привет, как дела? \U0001f426" in json.dumps(
            payloads[-1], ensure_ascii=False
        )
        assert "/status" not in json.dumps(payloads[-1])
        assert "Synthetic persona" in json.dumps(payloads[-1])
        if with_life:
            assert "LIFE_PLAN" in "".join(lines)
            assert "DAILY_DIARY" in "".join(lines)
            assert "разобрал файлы" in json.dumps(payloads[-1], ensure_ascii=False)
        live = History.load(tmp_path / "runtime/chat_history.json")
        assert [m.text for m in live.messages] == [
            "Привет, как дела? \U0001f426",
            reply,
        ]
    finally:
        release.set()
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
        if process.stdin is not None:
            process.stdin.close()
        reader.join(timeout=2)
        if process.stdout is not None:
            process.stdout.close()
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=2)
