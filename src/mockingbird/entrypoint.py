from __future__ import annotations

import argparse
import io
import sys
from collections.abc import Sequence
from pathlib import Path

from mockingbird.config import CONFIG_PATH, load_config
from mockingbird.interfaces.telegram.options import TelegramOptions
from mockingbird.interfaces.terminal.options import TerminalOptions

TELEGRAM_COMMANDS = (
    "status",
    "observe",
    "export",
    "prepare-test",
    "send-test",
    "confirm-appearance",
    "resolve-delivery",
    "run",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mockingbird",
        description="Build and run a persona through the terminal or Telegram Business.",
        epilog=(
            "Examples: mockingbird --interface cli --fast-replies; "
            "mockingbird --interface telegram run --fast-replies; "
            "mockingbird --interface telegram status --live"
        ),
    )
    parser.add_argument(
        "--interface",
        choices=("cli", "telegram"),
        default="cli",
        help="Choose the local terminal or Telegram Business interface (default: cli).",
    )
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--state-dir", type=Path)
    parser.add_argument("--seed", type=int)
    refresh = parser.add_mutually_exclusive_group()
    refresh.add_argument("--refresh-persona", action="store_true")
    refresh.add_argument("--no-refresh-persona", action="store_true")
    parser.add_argument(
        "--fast-replies",
        action="store_true",
        help="Use immediate-mode scheduling for replies; initiatives keep their usual rules.",
    )
    parser.add_argument("--max-generations", type=int)

    parser.add_argument("command", nargs="?", choices=TELEGRAM_COMMANDS)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--seconds", type=int)
    parser.add_argument("--recipient", type=int)
    parser.add_argument("--text")
    parser.add_argument("--approval-id")
    parser.add_argument("--confirm-text")
    parser.add_argument("--message-id", type=int)
    parser.add_argument("--request-id")
    parser.add_argument("--outcome", choices=("sent", "not-sent"))
    return parser


def _options(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> TerminalOptions | TelegramOptions:
    if args.max_generations is not None and args.max_generations < 1:
        parser.error("--max-generations must be positive")

    if args.interface == "cli":
        if args.command is not None or any(
            (
                args.live,
                args.recipient is not None,
                args.text is not None,
                args.approval_id is not None,
                args.confirm_text is not None,
                args.message_id is not None,
                args.request_id is not None,
                args.outcome is not None,
                args.seconds is not None,
            )
        ):
            parser.error("Telegram command options require --interface telegram")
        return TerminalOptions(
            state_dir=args.state_dir,
            seed=args.seed,
            refresh_persona=args.refresh_persona,
            no_refresh_persona=args.no_refresh_persona,
            fast_replies=args.fast_replies,
            max_generations=args.max_generations,
        )

    if args.command is None:
        parser.error("--interface telegram requires a Telegram command")
    if args.seed is not None or args.refresh_persona or args.no_refresh_persona:
        parser.error("--seed and persona refresh options apply only to --interface cli")
    if args.command != "run" and (
        args.fast_replies or args.max_generations is not None
    ):
        parser.error("--fast-replies and --max-generations apply only to Telegram run")
    if args.command != "status" and args.live:
        parser.error("--live applies only to Telegram status")
    if args.command != "observe" and args.seconds is not None:
        parser.error("--seconds applies only to Telegram observe")
    if args.command == "prepare-test" and (args.recipient is None or args.text is None):
        parser.error("prepare-test requires --recipient and --text")
    if args.command == "send-test" and (
        args.approval_id is None or args.confirm_text is None
    ):
        parser.error("send-test requires --approval-id and --confirm-text")
    if args.command == "confirm-appearance" and args.message_id is None:
        parser.error("confirm-appearance requires --message-id")
    if args.command == "resolve-delivery" and (
        args.request_id is None or args.outcome is None
    ):
        parser.error("resolve-delivery requires --request-id and --outcome")
    if args.command != "prepare-test" and (
        args.recipient is not None or args.text is not None
    ):
        parser.error("--recipient and --text apply only to prepare-test")
    if args.command != "send-test" and (
        args.approval_id is not None or args.confirm_text is not None
    ):
        parser.error("--approval-id and --confirm-text apply only to send-test")
    if args.command not in ("confirm-appearance", "resolve-delivery") and (
        args.message_id is not None
    ):
        parser.error(
            "--message-id applies only to confirm-appearance or resolve-delivery"
        )
    if args.command != "resolve-delivery" and (
        args.request_id is not None or args.outcome is not None
    ):
        parser.error("--request-id and --outcome apply only to resolve-delivery")

    return TelegramOptions(
        state_dir=args.state_dir,
        command=args.command,
        live=args.live,
        seconds=args.seconds if args.seconds is not None else 60,
        recipient=args.recipient,
        text=args.text,
        approval_id=args.approval_id,
        confirm_text=args.confirm_text,
        message_id=args.message_id,
        request_id=args.request_id,
        outcome=args.outcome,
        fast_replies=args.fast_replies,
        max_generations=args.max_generations,
    )


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    options = _options(parser, args)
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        if isinstance(stream, io.TextIOWrapper):
            stream.reconfigure(encoding="utf-8")
    try:
        config = load_config(args.config)
        if isinstance(options, TerminalOptions):
            from mockingbird.interfaces.terminal.interface import TerminalInterface

            TerminalInterface(options).run(config)
        else:
            from mockingbird.interfaces.telegram.interface import TelegramInterface

            TelegramInterface(options).run(config)
    except SystemExit:
        raise
    except (ValueError, RuntimeError, OSError, KeyError) as exc:
        if isinstance(options, TerminalOptions):
            parser.exit(1, f"Не удалось продолжить чат: {exc}\n")
        parser.exit(
            1,
            f"Business остановлен: {type(exc).__name__}. Проверь локальные настройки.\n",
        )
    except KeyboardInterrupt:
        parser.exit(130, "Mockingbird остановлен.\n")


if __name__ == "__main__":
    main()
