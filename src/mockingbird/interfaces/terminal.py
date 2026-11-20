from __future__ import annotations

import argparse
import io
import queue
import sys
import threading
import time
from contextlib import ExitStack
from datetime import datetime
from pathlib import Path
from typing import BinaryIO

from mockingbird.chat.events import PipelineEvent
from mockingbird.chat.pipeline import ChatPipeline
from mockingbird.chat.presenter import reason_label
from mockingbird.config import CONFIG_PATH, load_config
from mockingbird.conversation.identity import pair_id
from mockingbird.llm.client import LLM
from mockingbird.mockingbird.mockingbird import Mockingbird
from mockingbird.persona.refresh import refresh_needed, refresh_persona

if sys.platform == "win32":
    import msvcrt
else:
    import fcntl


class SessionLock:
    """OS lock released automatically on process exit, including a crash."""

    def __init__(self, directory: Path):
        self.path = directory / ".session.lock"
        self.stream: BinaryIO | None = None

    def __enter__(self) -> SessionLock:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        stream = self.path.open("a+b")
        if self.path.stat().st_size == 0:
            stream.write(b" ")
            stream.flush()
        stream.seek(0)
        try:
            if sys.platform == "win32":
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            stream.close()
            raise RuntimeError(
                "Эта папка состояния уже открыта другим процессом. Выбери другой --state-dir."
            ) from exc
        self.stream = stream
        return self

    def __exit__(self, *args: object) -> None:
        if self.stream is not None:
            if sys.platform == "win32":
                self.stream.seek(0)
                msvcrt.locking(self.stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(self.stream.fileno(), fcntl.LOCK_UN)
            self.stream.close()
            self.stream = None


def render(event: PipelineEvent) -> None:
    stamp = datetime.fromisoformat(event.at).strftime("%H:%M:%S")
    print(f"[{stamp}] {event.stage} | {event.message}", flush=True)
    if "text" in event.data:
        print(str(event.data["text"]), flush=True)
    if "reason" in event.data:
        print(
            "  " + reason_label(event.data) + ": " + str(event.data["reason"]),
            flush=True,
        )
    if event.stage == "REQUEST" and event.data.get("request_path"):
        print("  Контекст: " + str(event.data["request_path"]), flush=True)


def run_console(pipeline: ChatPipeline) -> None:
    inbox: queue.Queue[tuple[datetime, str | None]] = queue.Queue()

    def read_input() -> None:
        while True:
            try:
                text = input()
            except EOFError:
                inbox.put((pipeline.clock(), None))
                return
            inbox.put((pipeline.clock(), text))
            if text.strip() == "/quit":
                return

    try:
        pipeline.start()
        print(
            "Пиши сообщение и нажимай Enter. /help — команды, /quit — выход.",
            flush=True,
        )
        threading.Thread(
            target=read_input, daemon=True, name="mockingbird-input"
        ).start()
        while True:
            # Drain input before consuming completed model results, so a message
            # arriving during generation invalidates that result before output.
            while True:
                try:
                    received_at, text = inbox.get_nowait()
                except queue.Empty:
                    break
                if text is None:
                    pipeline.emit(
                        "EOF",
                        "Ввод закрыт. Завершаем чат; ожидающие ответы сохранены для перезапуска.",
                    )
                    return
                if text.startswith("/") and not text.startswith("//"):
                    if not pipeline.command(text):
                        return
                else:
                    pipeline.receive(
                        text[1:] if text.startswith("//") else text, received_at
                    )
            pipeline.step()
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        pipeline.close()


def main() -> None:
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        if isinstance(stream, io.TextIOWrapper):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(
        description="Наблюдаемый терминальный чат с самостоятельными инициативами."
    )
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--state-dir", type=Path)
    parser.add_argument("--seed", type=int)
    refresh_flags = parser.add_mutually_exclusive_group()
    refresh_flags.add_argument(
        "--refresh-persona",
        action="store_true",
        help="Three building_model requests: rebuild profile, Life Prompt, and initial diary from recent messages; previous versions are saved",
    )
    refresh_flags.add_argument(
        "--no-refresh-persona",
        action="store_true",
        help="Не вызывать building_model при запуске, использовать текущие артефакты",
    )
    parser.add_argument(
        "--fast-replies",
        action="store_true",
        help="Ответы через короткую паузу, даже вне активных часов; инициативы по истории",
    )
    parser.add_argument(
        "--max-generations",
        type=int,
        help="Остановить новые запросы после указанного количества за этот запуск",
    )
    args = parser.parse_args()
    if args.max_generations is not None and args.max_generations < 1:
        parser.error("--max-generations must be positive")
    try:
        print(
            "START | Загружаем настройки, историю, профиль и дневник; строим расписание по истории...",
            flush=True,
        )
        config = load_config(args.config)
        if args.fast_replies:
            config.setdefault("scheduler", {})["reply_mode"] = "immediate"
        identity = pair_id(config["user"], config["other"])
        paths = config["paths"]
        processed = Path(paths["processed"])
        runtime = Path(paths.get("runtime", processed.parent / "runtime"))
        memory_root = Path(paths.get("memory", runtime / "memory"))
        base = runtime
        if config.get("scheduler", {}).get("reply_mode") == "immediate":
            base /= "fast-replies"
        directory = args.state_dir or base / identity
        startup_generations = 0
        with ExitStack() as locks:
            locks.enter_context(SessionLock(directory))
            if config.get("diary", {}).get("enabled", True) or config.get(
                "life", {}
            ).get("enabled", False):
                memory_directory = memory_root / identity
                locks.enter_context(SessionLock(memory_directory))
            should_refresh = args.refresh_persona or (
                not args.no_refresh_persona
                and config.get("persona_build", {}).get("auto_refresh_on_start", False)
                and refresh_needed(config)
            )
            if should_refresh:
                if args.max_generations is not None and args.max_generations < 3:
                    raise ValueError(
                        "Persona refresh requires three requests. Limit is too low; use --no-refresh-persona or --max-generations of at least 3."
                    )
                settings = config["building_model"]
                with SessionLock(processed):
                    refresh_persona(
                        config,
                        LLM(
                            settings["host"],
                            settings["model"],
                            settings.get("api_key"),
                            settings.get("temperature", 0.3),
                        ),
                    )
                startup_generations = 3
            bird = Mockingbird.from_config(config, args.seed)
            bird.runtime_store.restore(directory)
            if bird.memory is not None:
                imported = bird.memory.import_sessions(runtime)
                bird.memory.render()
                bird.diary = bird.memory.combined()
                print(
                    f"MEMORY_RESTORE | В очереди {len(bird.memory.pending)} входящих, новых из других сессий: {imported}. Ответы модели не используются как факты.",
                    flush=True,
                )
            settings = config["inference_model"]
            pipeline = ChatPipeline(
                bird,
                directory,
                render,
                model_info={
                    key: settings[key]
                    for key in ("host", "model", "temperature", "json_mode")
                    if key in settings
                },
                max_generations=args.max_generations,
            )
            pipeline.generations = startup_generations
            run_console(pipeline)
    except (ValueError, RuntimeError, OSError, KeyError) as exc:
        parser.exit(1, f"Не удалось продолжить чат: {exc}\n")


if __name__ == "__main__":
    main()
