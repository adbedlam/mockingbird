from pathlib import Path

import pytest

from mockingbird import entrypoint
from mockingbird.interfaces.telegram import interface
from mockingbird.interfaces.telegram.options import TelegramOptions
from mockingbird.interfaces.terminal import interface
from mockingbird.interfaces.terminal.options import TerminalOptions


def parse_options(*argv: str) -> TerminalOptions | TelegramOptions:
    parser = entrypoint.build_parser()
    return entrypoint._options(parser, parser.parse_args(argv))


def test_terminal_flags_become_typed_options(tmp_path: Path):
    state_dir = tmp_path / "runtime"
    options = parse_options(
        "--interface",
        "cli",
        "--state-dir",
        str(state_dir),
        "--fast-replies",
        "--max-generations",
        "3",
    )
    assert isinstance(options, TerminalOptions)
    assert options.state_dir == state_dir
    assert options.fast_replies
    assert options.max_generations == 3


def test_telegram_run_flags_become_typed_options():
    options = parse_options(
        "--interface",
        "telegram",
        "run",
        "--fast-replies",
        "--max-generations",
        "2",
    )
    assert isinstance(options, TelegramOptions)
    assert options.command == "run"
    assert options.fast_replies
    assert options.max_generations == 2


@pytest.mark.parametrize(
    "argv",
    [
        ("--interface", "telegram"),
        ("--interface", "telegram", "prepare-test"),
        ("--interface", "cli", "run"),
        ("--interface", "telegram", "status", "--fast-replies"),
        ("--interface", "telegram", "status", "--seconds", "60"),
    ],
)
def test_interface_rejects_missing_or_inapplicable_flags(argv: tuple[str, ...]):
    parser = entrypoint.build_parser()
    with pytest.raises(SystemExit):
        entrypoint._options(parser, parser.parse_args(argv))


@pytest.mark.parametrize(
    ("interface", "command", "module", "class_name"),
    [
        ("cli", [], "mockingbird.interfaces.terminal.interface", "TerminalInterface"),
        (
            "telegram",
            ["run"],
            "mockingbird.interfaces.telegram.interface",
            "TelegramInterface",
        ),
    ],
)
def test_main_dispatches_to_selected_interface(
    monkeypatch, interface: str, command: list[str], module: str, class_name: str
):
    received = []

    class FakeInterface:
        def __init__(self, options):
            received.append(options)

        def run(self, config):
            received.append(config)

    monkeypatch.setattr(entrypoint, "load_config", lambda path: {"loaded": True})
    interface_module = __import__(module, fromlist=[class_name])
    monkeypatch.setattr(interface_module, class_name, FakeInterface)

    entrypoint.main(["--interface", interface, *command])

    expected_type = TerminalOptions if interface == "cli" else TelegramOptions
    assert isinstance(received[0], expected_type)
    assert received[1] == {"loaded": True}


def test_interfaces_do_not_export_cli_entrypoints():
    assert not hasattr(interface, "main")
    assert not hasattr(interface, "main")
