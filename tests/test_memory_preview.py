import json
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import yaml
from test_local_persona import config_for

from mockingbird.persona.diary.diary import Diary
from mockingbird.persona.diary.runtime import DiaryMemory


def test_memory_preview_cli_prepare_generate_resume_and_preserved_requests(tmp_path):
    config = config_for(tmp_path)
    payloads = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            payloads.append(payload)
            body = json.loads(payload["messages"][-1]["content"])
            evidence = body["incoming_evidence"][0]
            item = {"timestamp": evidence["timestamp"], "source_ids": [evidence["id"]]}
            if len(payloads) == 1:
                item.update(
                    op="add",
                    content="Partner plans project tests",
                    subject="partner",
                    topic="project.tests",
                    state="planned",
                )
            else:
                target = body["recent_existing_diary"][0]
                item.update(
                    op="confirm" if len(payloads) == 2 else "update",
                    target_id=target["memory_id"],
                    target_revision=target["revision"],
                )
                if len(payloads) == 3:
                    item.update(content="Partner finished the tests", state="completed")
            raw = json.dumps({"operations": [item], "reason": "Visible batch decision"})
            response = json.dumps(
                {
                    "id": "test",
                    "object": "chat.completion",
                    "created": 0,
                    "model": "test",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": raw},
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
    output = tmp_path / "preview"
    base = [
        sys.executable,
        "scripts/preview_memory.py",
        "--config",
        str(config_path),
        "--output",
        str(output),
    ]
    env = {
        **os.environ,
        "PYTHONIOENCODING": "utf-8",
        "INFERENCE_MODEL_API_KEY": "test-only",
        "NO_PROXY": "127.0.0.1,localhost",
    }

    def run(args):
        return subprocess.run(
            base + args,
            cwd=Path(__file__).resolve().parents[1],
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=20,
        )

    try:
        prepared = run(["--prepare", "--at", "2026-11-17T13:00:00+03:00"])
        assert prepared.returncode == 0, prepared.stdout + prepared.stderr
        assert len(payloads) == 0
        first_request = (output / "1.request.json").read_bytes()
        generated = run(["--generate", "--resume"])
        assert generated.returncode == 0, generated.stdout + generated.stderr
        assert "New API calls: 3" in generated.stdout
        assert len(payloads) == 3
        assert all("PARTNER_ONLY" not in json.dumps(payload) for payload in payloads)
        assert (output / "1.request.json").read_bytes() == first_request
        memory = DiaryMemory(output / "memory", "Persona", "Partner", Diary())
        assert len(memory.entries) == 1
        assert (
            memory.entries[0].revision == 2 and memory.entries[0].state == "completed"
        )
        assert len(memory.versions[str(memory.entries[0].memory_id)]) == 3
        assert "Partner plans" in (output / "memory/diary_history.md").read_text(
            encoding="utf-8"
        )
        replay = run(["--generate", "--resume"])
        assert replay.returncode == 0, replay.stdout + replay.stderr
        assert "New API calls: 0" in replay.stdout
        assert len(payloads) == 3
        wrong_date = run(
            ["--generate", "--resume", "--at", "2026-11-18T13:00:00+03:00"]
        )
        assert wrong_date.returncode != 0
        assert "original virtual starting time" in wrong_date.stderr
        assert len(payloads) == 3
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=2)
