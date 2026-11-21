from __future__ import annotations

import queue
import threading
import time
from contextlib import ExitStack
from datetime import datetime
from pathlib import Path

from mockingbird.chat.chat import Chat
from mockingbird.conversation.identity import pair_id
from mockingbird.interfaces.common import SessionLock, render
from mockingbird.interfaces.terminal.options import TerminalOptions
from mockingbird.llm.client import LLM
from mockingbird.mockingbird.mockingbird import Mockingbird
from mockingbird.persona.refresh import refresh_needed, refresh_persona


def run_console(chat: Chat) -> None:
    inbox = _create_inbox()

    try:
        chat.start()
        _print_console_help()
        _start_input_thread(chat, inbox)
        _console_loop(chat, inbox)

    except KeyboardInterrupt:
        pass

    finally:
        chat.close()


def _create_inbox() -> queue.Queue[tuple[datetime, str | None]]:
    return queue.Queue()


def _print_console_help() -> None:
    print(
        "Пиши сообщение и нажимай Enter. /help — команды, /quit — выход.",
        flush=True,
    )


def _start_input_thread(
    chat: Chat,
    inbox: queue.Queue[tuple[datetime, str | None]],
) -> None:
    threading.Thread(
        target=_read_input,
        args=(chat, inbox),
        daemon=True,
        name="mockingbird-input",
    ).start()


def _read_input(
    chat: Chat,
    inbox: queue.Queue[tuple[datetime, str | None]],
) -> None:
    while True:
        try:
            text = input()
        except EOFError:
            inbox.put((chat.clock(), None))
            return

        inbox.put((chat.clock(), text))

        if text.strip() == "/quit":
            return


def _console_loop(
    chat: Chat,
    inbox: queue.Queue[tuple[datetime, str | None]],
) -> None:
    while True:
        if not _process_inbox(chat, inbox):
            return

        chat.step()
        time.sleep(0.1)


def _process_inbox(
    chat: Chat,
    inbox: queue.Queue[tuple[datetime, str | None]],
) -> bool:
    """Process all queued input. Return False when the console should exit."""
    while True:
        try:
            received_at, text = inbox.get_nowait()
        except queue.Empty:
            return True

        if text is None:
            _handle_eof(chat)
            return False

        if not _process_input(chat, text, received_at):
            return False


def _process_input(
    chat: Chat,
    text: str,
    received_at: datetime,
) -> bool:
    if text.startswith("/") and not text.startswith("//"):
        return _process_command(chat, text)

    message = text[1:] if text.startswith("//") else text
    chat.receive(message, received_at)
    return True


def _process_command(chat: Chat, command: str) -> bool:
    return chat.command(command)


def _handle_eof(chat: Chat) -> None:
    chat.emit(
        "EOF",
        "Ввод закрыт. Завершаем чат; ожидающие ответы сохранены для перезапуска.",
    )


class TerminalInterface:
    def __init__(self, options: TerminalOptions):
        self.options = options

    def run(self, config: dict) -> None:
        options = self.options

        self._print_startup()

        self._configure_reply_mode(config)
        identity = pair_id(config["user"], config["other"])

        directory, memory_directory, processed = self._resolve_paths(config, identity)

        with self._acquire_locks(config, directory, memory_directory):
            startup_generations = self._refresh_persona(config, processed)

            mockingbird = Mockingbird.from_config(config, seed=options.seed)
            mockingbird.runtime_store.restore(directory)

            self._restore_memory(mockingbird, self._runtime_path(config))

            chat = self._create_chat(mockingbird, directory, config)
            chat.generations = startup_generations

            run_console(chat)

    def _print_startup(self) -> None:
        print(
            "START | Загружаем настройки, историю, профиль и дневник; строим расписание по истории...",
            flush=True,
        )

    def _configure_reply_mode(self, config: dict) -> None:
        if self.options.fast_replies:
            config.setdefault("scheduler", {})["reply_mode"] = "immediate"

    def _runtime_path(self, config: dict) -> Path:
        paths = config["paths"]
        processed = Path(paths["processed"])
        return Path(paths.get("runtime", processed.parent / "runtime"))

    def _resolve_paths(self, config: dict, identity: str) -> tuple[Path, Path, Path]:
        paths = config["paths"]

        processed = Path(paths["processed"])
        runtime = self._runtime_path(config)
        memory_root = Path(paths.get("memory", runtime / "memory"))

        base = runtime
        if config.get("scheduler", {}).get("reply_mode") == "immediate":
            base /= "fast-replies"

        directory = self.options.state_dir or base / identity
        memory_directory = memory_root / identity

        return directory, memory_directory, processed

    def _acquire_locks(
        self,
        config: dict,
        directory: Path,
        memory_directory: Path,
    ) -> ExitStack:
        locks = ExitStack()
        locks.enter_context(SessionLock(directory))

        if config.get("diary", {}).get("enabled", True) or config.get("life", {}).get(
            "enabled", False
        ):
            locks.enter_context(SessionLock(memory_directory))

        return locks

    def _refresh_persona(self, config: dict, processed: Path) -> int:
        options = self.options

        should_refresh = options.refresh_persona or (
            not options.no_refresh_persona
            and config.get("persona_build", {}).get("auto_refresh_on_start", False)
            and refresh_needed(config)
        )

        if not should_refresh:
            return 0

        if options.max_generations is not None and options.max_generations < 3:
            raise ValueError(
                "Persona refresh requires three requests. Limit is too low; "
                "use --no-refresh-persona or --max-generations of at least 3."
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

        return 3

    def _restore_memory(self, mockingbird: Mockingbird, runtime: Path) -> None:
        memory = mockingbird.memory

        if memory is None:
            return

        imported = memory.import_sessions(runtime)
        memory.render()
        mockingbird.diary = memory.combined()

        print(
            f"MEMORY_RESTORE | В очереди {len(memory.pending)} входящих, "
            f"новых из других сессий: {imported}. "
            "Ответы модели не используются как факты.",
            flush=True,
        )

    def _create_chat(
        self,
        mockingbird: Mockingbird,
        directory: Path,
        config: dict,
    ) -> Chat:
        settings = config["inference_model"]

        return Chat(
            mockingbird,
            directory,
            render,
            model_info={
                key: settings[key]
                for key in ("host", "model", "temperature", "json_mode")
                if key in settings
            },
            max_generations=self.options.max_generations,
        )
