from __future__ import annotations

import queue
import threading
import time
from contextlib import ExitStack
from datetime import datetime
from pathlib import Path

from mockingbird.chat.pipeline import ChatPipeline
from mockingbird.chat.session import create_bird, create_pipeline
from mockingbird.conversation.identity import pair_id
from mockingbird.interfaces.common import SessionLock, render
from mockingbird.interfaces.terminal.options import TerminalOptions
from mockingbird.llm.client import LLM
from mockingbird.persona.refresh import refresh_needed, refresh_persona


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


class TerminalInterface:
    def __init__(self, options: TerminalOptions):
        self.options = options

    def run(self, config: dict) -> None:
        options = self.options
        print(
            "START | Загружаем настройки, историю, профиль и дневник; строим расписание по истории...",
            flush=True,
        )
        if options.fast_replies:
            config.setdefault("scheduler", {})["reply_mode"] = "immediate"
        identity = pair_id(config["user"], config["other"])
        paths = config["paths"]
        processed = Path(paths["processed"])
        runtime = Path(paths.get("runtime", processed.parent / "runtime"))
        memory_root = Path(paths.get("memory", runtime / "memory"))
        base = runtime
        if config.get("scheduler", {}).get("reply_mode") == "immediate":
            base /= "fast-replies"
        directory = options.state_dir or base / identity
        startup_generations = 0
        with ExitStack() as locks:
            locks.enter_context(SessionLock(directory))
            if config.get("diary", {}).get("enabled", True) or config.get(
                "life", {}
            ).get("enabled", False):
                memory_directory = memory_root / identity
                locks.enter_context(SessionLock(memory_directory))
            should_refresh = options.refresh_persona or (
                not options.no_refresh_persona
                and config.get("persona_build", {}).get("auto_refresh_on_start", False)
                and refresh_needed(config)
            )
            if should_refresh:
                if options.max_generations is not None and options.max_generations < 3:
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
            bird = create_bird(config, seed=options.seed)
            bird.runtime_store.restore(directory)
            if bird.memory is not None:
                imported = bird.memory.import_sessions(runtime)
                bird.memory.render()
                bird.diary = bird.memory.combined()
                print(
                    f"MEMORY_RESTORE | В очереди {len(bird.memory.pending)} входящих, новых из других сессий: {imported}. Ответы модели не используются как факты.",
                    flush=True,
                )
            pipeline = create_pipeline(
                config,
                bird,
                directory,
                render,
                max_generations=options.max_generations,
            )
            pipeline.generations = startup_generations
            run_console(pipeline)
