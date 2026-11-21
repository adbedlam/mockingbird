import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from tests.config_helpers import scheduler_settings

from mockingbird.conversation.history import History
from mockingbird.conversation.message import Message
from mockingbird.persona.diary.diary import Diary
from mockingbird.persona.inner.inner import InnerPrompt
from mockingbird.scheduler.profile import BehaviorProfile
from mockingbird.scheduler.scheduler import Action, Scheduler


@pytest.mark.parametrize("prepare", [False, True])
def test_preview_writes_report_without_network_or_artifact_changes(
    tmp_path: Path, prepare: bool
):
    from datetime import datetime, timedelta

    root = Path(__file__).resolve().parents[2]
    processed = tmp_path / "processed"
    messages = []
    for offset in range(14):
        day = datetime(2026, 10, 1) + timedelta(days=offset)
        messages.extend(
            [
                Message("Partner", day.replace(hour=8), text="hi"),
                Message("Persona", day.replace(hour=9), text="hello"),
            ]
        )
    History("Persona", "Partner", messages).save(processed / "history.json")
    InnerPrompt("Synthetic profile").save(processed / "inner.md")
    Diary().save(processed / "diary.json")
    before = {p.name: p.read_bytes() for p in processed.iterdir()}
    config = {
        "user": "Persona",
        "other": "Partner",
        # No real key: constructing an LLM or making a network request is a bug.
        "inference_model": {"host": "https://example.invalid/v1", "model": "fake"},
        "scheduler": scheduler_settings(),
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
    output = tmp_path / "report.md"
    command = [
        sys.executable,
        str(root / "scripts/preview_scheduler.py"),
        "--config",
        str(config_path),
        "--date",
        "2026-11-17",
        "--days",
        "2",
        "--output",
        str(output),
    ]
    if prepare:
        command.append("--prepare-request")
    result = subprocess.run(command, cwd=root, capture_output=True, timeout=20)
    assert result.returncode == 0, result.stderr.decode(errors="replace")
    assert output.exists()
    events = json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))[
        "events"
    ]
    assert any(
        e["action"] == ("REQUEST_PREPARED" if prepare else "SEND initiative")
        for e in events
    )
    assert {p.name: p.read_bytes() for p in processed.iterdir()} == before
    assert not (tmp_path / "runtime").exists()
    if prepare:
        payload = json.loads(
            output.with_suffix(".request.json").read_text(encoding="utf-8")
        )
        assert payload["endpoint"] == "https://example.invalid/v1/chat/completions"
        assert payload["body"]["model"] == "fake"
        assert payload["body"]["messages"]
        assert "api_key" not in json.dumps(payload)


def test_slow_generation_cannot_send_outside_active_window():
    from datetime import datetime
    from zoneinfo import ZoneInfo

    zone = ZoneInfo("Europe/Moscow")
    now = datetime(2026, 11, 17, 9, tzinfo=zone)
    history = History(
        "Persona",
        "Partner",
        [Message("Partner", now.replace(hour=8)), Message("Persona", now)],
    )
    scheduler = Scheduler(BehaviorProfile(history, scheduler_settings()))
    scheduler.day = now.date()
    scheduler.planned = [now]
    decision = scheduler.tick(now)
    assert decision.action == Action.SEND
    with pytest.raises(ValueError, match="send window expired"):
        scheduler.mark_sent(decision, now.replace(hour=10))
    assert scheduler.initiations_sent == 0


def test_zero_daily_budget_disables_initiatives():
    from datetime import datetime
    from zoneinfo import ZoneInfo

    now = datetime(2026, 11, 17, 9, tzinfo=ZoneInfo("Europe/Moscow"))
    history = History(
        "Persona",
        "Partner",
        [Message("Partner", now.replace(hour=8)), Message("Persona", now)],
    )
    scheduler = Scheduler(
        BehaviorProfile(history, scheduler_settings(max_initiations_per_day=0))
    )
    assert scheduler.tick(now).action == Action.WAIT
    assert not scheduler.planned
