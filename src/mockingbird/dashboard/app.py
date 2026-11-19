"""Read-only local Persona dashboard server and data API."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
import webbrowser
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import httpx

from mockingbird.config import CONFIG_PATH, PROJECT_ROOT, load_config
from mockingbird.conversation.identity import pair_id
from mockingbird.observability import redact


def read_json(path: Path, default: object = None) -> object:
    if not path.is_file() or path.is_symlink() or path.stat().st_size > 12_000_000:
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return default


def tail_events(path: Path, count: int = 350) -> list[dict]:
    if not path.is_file() or path.is_symlink():
        return []
    with path.open("rb") as stream:
        size = path.stat().st_size
        offset = max(0, size - 2_000_000)
        stream.seek(offset)
        raw = stream.read(2_000_000)
    lines = raw.decode("utf-8", errors="replace").splitlines()
    if offset:
        lines = lines[1:]
    events = []
    for line in lines[-count:]:
        try:
            row = json.loads(line)
            if isinstance(row, dict):
                events.append(row)
        except ValueError:
            continue  # An append may be in progress. The next refresh retries it.
    return events


class DashboardData:
    def __init__(
        self,
        runtime: Path,
        memory: Path,
        processed: Path,
        identity: str,
        user: str,
        other: str,
    ):
        self.runtime, self.memory, self.processed = (
            runtime.resolve(),
            memory.resolve(),
            processed.resolve(),
        )
        self.identity, self.user, self.other = identity, user, other

    def sessions(self) -> list[dict]:
        result: list[dict] = []
        if not self.runtime.is_dir():
            return result
        for directory in self.runtime.rglob(self.identity):
            if (
                not directory.is_dir()
                or directory.is_symlink()
                or not directory.resolve().is_relative_to(self.runtime)
            ):
                continue
            files = [
                p for p in directory.iterdir() if p.is_file() and not p.is_symlink()
            ]
            if not files:
                continue
            relative = directory.relative_to(self.runtime).as_posix()
            saved = read_json(directory / "runtime_state.json", {})
            legacy = read_json(directory / "chat_history.json", {})
            has_messages = isinstance(saved, dict) and bool(
                saved.get("history", {}).get("messages")
            )
            has_messages = has_messages or (
                isinstance(legacy, dict) and bool(legacy.get("messages"))
            )
            result.append(
                {
                    "id": hashlib.sha256(relative.encode()).hexdigest()[:16],
                    "label": relative,
                    "has_messages": has_messages,
                    "updated": max(p.stat().st_mtime for p in files),
                }
            )
        return sorted(result, key=lambda row: row["updated"], reverse=True)

    def directory(self, identity: str) -> Path:
        session = next((row for row in self.sessions() if row["id"] == identity), None)
        if session is None:
            raise ValueError("Unknown session")
        return self.runtime / str(session["label"])

    def request(self, session_id: str, request_id: str) -> object:
        if len(request_id) != 32 or any(
            c not in "0123456789abcdef" for c in request_id
        ):
            raise ValueError("Invalid request ID")
        return read_json(
            self.directory(session_id) / "requests" / f"{request_id}.json", {}
        )

    def snapshot(self, session_id: str | None) -> dict:
        sessions = self.sessions()
        preferred = next(
            (row for row in sessions if row["has_messages"]),
            sessions[0] if sessions else None,
        )
        selected = session_id or (str(preferred["id"]) if preferred else None)
        directory = self.directory(selected) if selected else None

        def read(path: Path, default):
            return read_json(path, default)

        state = read(directory / "runtime_state.json", {}) if directory else {}
        history = state.get("history") or (
            read(directory / "chat_history.json", {}) if directory else {}
        )
        scheduler = state.get("scheduler") or (
            read(directory / "scheduler.json", {}) if directory else {}
        )
        memory = read(self.memory / "memory.json", {})
        seed = read(self.processed / "diary.json", {})
        life = read(self.memory / "life" / "life.json", {})
        events = tail_events(directory / "decisions.jsonl") if directory else []
        requests = []
        if directory:
            paths = sorted(
                (directory / "requests").glob("*.json"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            for path in paths[:40]:
                row = read(path, {})
                if not isinstance(row, dict):
                    continue
                requests.append(
                    {
                        k: row.get(k)
                        for k in (
                            "id",
                            "at",
                            "kind",
                            "status",
                            "elapsed_seconds",
                            "error",
                            "decision",
                            "run_id",
                        )
                    }
                )
        daily_path = self.memory / "life" / "daily_diary.md"
        daily = (
            daily_path.read_text(encoding="utf-8")
            if daily_path.is_file()
            and not daily_path.is_symlink()
            and daily_path.stat().st_size < 2_000_000
            else ""
        )
        business = read(directory / "business_status.json", {}) if directory else {}
        deliveries = read(directory / "delivery_report.json", {}) if directory else {}
        if business.get("recent_deliveries") is not None:
            deliveries = {**deliveries, "outbox": business["recent_deliveries"]}
        return {
            "at": datetime.now(timezone.utc).isoformat(),
            "sessions": sessions,
            "session_id": selected,
            "user": self.user,
            "other": self.other,
            "events": events,
            "history": history.get("messages", [])[-250:],
            "scheduler": scheduler,
            "requests": requests,
            "memory": memory,
            "seed_diary": seed,
            "life": life,
            "daily_diary": daily,
            "business": business,
            "deliveries": deliveries,
            "limits": {"events": 350, "messages": 250, "requests": 40},
            "note": "Пояснения решений модели; скрытая цепочка рассуждений API недоступна.",
        }


def create_server(
    data: DashboardData, key: str, port: int = 8765
) -> ThreadingHTTPServer:
    html = Path(__file__).with_name("dashboard.html").read_bytes()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args: object) -> None:
            pass  # Do not put query credentials or conversation text in access logs.

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            query = parse_qs(parsed.query)
            authorized = (
                self.headers.get("X-Dashboard-Key") or query.get("key", [""])[0]
            )
            if not secrets.compare_digest(authorized, key):
                self.send_error(403)
                return
            host = self.headers.get("Host", "").split(":")[0]
            if host not in ("127.0.0.1", "localhost"):
                self.send_error(403)
                return
            try:
                if parsed.path == "/":
                    payload, kind = html, "text/html; charset=utf-8"
                elif parsed.path == "/api/state":
                    payload = json.dumps(
                        data.snapshot(query.get("session", [None])[0]),
                        ensure_ascii=False,
                    ).encode()
                    kind = "application/json; charset=utf-8"
                elif parsed.path == "/api/request":
                    payload = json.dumps(
                        data.request(
                            query.get("session", [""])[0], query.get("id", [""])[0]
                        ),
                        ensure_ascii=False,
                    ).encode()
                    kind = "application/json; charset=utf-8"
                elif parsed.path == "/health":
                    payload = json.dumps(
                        {"app": "mockingbird-dashboard", "pair_id": data.identity}
                    ).encode()
                    kind = "application/json; charset=utf-8"
                else:
                    self.send_error(404)
                    return
                payload = redact(payload.decode()).encode()
            except (ValueError, OSError, KeyError):
                self.send_error(400)
                return
            self.send_response(200)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; connect-src 'self'; img-src 'self'; frame-ancestors 'none'; base-uri 'none'",
            )
            self.end_headers()
            self.wfile.write(payload)

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def dashboard_key(path: Path) -> str:
    """Keep localhost links valid across process restarts. Never a Bot Token."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        key = path.read_text(encoding="utf-8").strip()
        if len(key) >= 24 and all(char.isalnum() or char in "-_" for char in key):
            return key
        raise ValueError("Dashboard access key is invalid; check data/dashboard")
    key = secrets.token_urlsafe(24)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(key)
    return key


def running_dashboard(port: int, key: str, identity: str) -> bool:
    try:
        with httpx.Client(trust_env=False, timeout=2) as client:
            response = client.get(
                f"http://127.0.0.1:{port}/health", headers={"X-Dashboard-Key": key}
            )
            return response.status_code == 200 and response.json() == {
                "app": "mockingbird-dashboard",
                "pair_id": identity,
            }
    except (httpx.HTTPError, ValueError):
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Локальный read-only дашборд Persona")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--open",
        action="store_true",
        help="Открыть актуальную локальную ссылку в браузере",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Проверить, запущен ли дашборд, без запуска сервера",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    identity = pair_id(config["user"], config["other"])
    processed = Path(config["paths"]["processed"])
    memory = Path(config["paths"].get("memory", processed.parent / "memory")) / identity
    data = DashboardData(
        PROJECT_ROOT / "data/runtime",
        memory,
        processed,
        identity,
        config["user"],
        config["other"],
    )
    if not 1 <= args.port <= 65535:
        parser.error("Порт должен быть от 1 до 65535")
    local_state = PROJECT_ROOT / "data/dashboard"
    key = dashboard_key(local_state / f"access-{args.port}.key")
    url = f"http://127.0.0.1:{args.port}/?key={key}"
    active = running_dashboard(args.port, key, identity)
    if args.status:
        print(
            json.dumps(
                {"running": active, "url": url, "pair_id": identity}, ensure_ascii=False
            )
        )
        return
    if active:
        print(f"Дашборд уже работает: {url}", flush=True)
        if args.open:
            webbrowser.open(url)
        return
    try:
        server = create_server(data, key, args.port)
    except OSError:
        parser.exit(
            1,
            f"Порт {args.port} занят другим процессом или недоступен. Попробуй --port {args.port + 1}.\n",
        )
    info = local_state / f"instance-{args.port}.json"
    info.write_text(
        json.dumps(
            {"pid": os.getpid(), "url": url, "pair_id": identity}, ensure_ascii=False
        ),
        encoding="utf-8",
    )
    print(
        f"Persona dashboard: {url}",
        flush=True,
    )
    if args.open:
        # The server socket is already listening before the browser starts.
        webbrowser.open(url)
    print(
        "Только чтение. Ctrl+C — остановить. Данные и ключи API не отправляются наружу.",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
