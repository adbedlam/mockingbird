import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pytest
import yaml
from config_helpers import scheduler_settings

from mockingbird.conversation.history import History
from mockingbird.conversation.message import Message
from mockingbird.mockingbird.mockingbird import Mockingbird
from mockingbird.persona.refresh import bootstrap_local_persona
from mockingbird.persona.source import load_manifest


def config_for(tmp_path):
    source = tmp_path / "source"
    History(
        "Nikita",
        "Vanya",
        [
            Message("Vanya", datetime(2026, 11, 17, 10), "PARTNER_ONLY"),
            Message("Nikita", datetime(2026, 11, 17, 11), "my own spelling no period"),
        ],
    ).save(source / "history.json")
    return {
        "user": "Nikita",
        "other": "Vanya",
        "paths": {
            "raw": str(tmp_path / "raw"),
            "processed": str(source),
            "history_filename": "history.json",
            "inner_prompt_filename": "inner.md",
            "diary_filename": "diary.json",
        },
        "inference_model": {
            "host": "https://invalid.example/v1",
            "model": "never-called",
        },
        "scheduler": scheduler_settings(),
        "life": {"enabled": True},
        "diary": {"enabled": False},
    }


def test_local_profile_uses_only_target_examples_without_api_and_keeps_diary_empty(
    tmp_path,
):
    config = config_for(tmp_path)
    result = bootstrap_local_persona(config)
    assert result["profile"]["method"] == "local_examples"
    assert result["profile"]["model"] is None
    text = (Path(config["paths"]["processed"]) / "inner.md").read_text(encoding="utf-8")
    assert "my own spelling no period" in text
    assert "PARTNER_ONLY" not in text
    diary = json.loads(
        (Path(config["paths"]["processed"]) / "diary.json").read_text(encoding="utf-8")
    )
    assert diary == []
    assert load_manifest(Path(config["paths"]["processed"])) == result
    with pytest.raises(FileExistsError):
        bootstrap_local_persona(config)


def test_reversed_pair_cannot_use_cached_history_of_other_persona(tmp_path):
    config = config_for(tmp_path)
    config["user"], config["other"] = config["other"], config["user"]
    with pytest.raises(ValueError, match="configured participants"):
        bootstrap_local_persona(config)


def test_offline_preview_cli_and_runtime_life_directory(tmp_path):
    config = config_for(tmp_path)
    bootstrap_local_persona(config)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    path = tmp_path / "fixture.json"
    path.write_text(
        json.dumps(
            {
                "date": "2026-11-17",
                "activities": [
                    {
                        "start": "10:00",
                        "end": "11:00",
                        "activity": "sorted project files",
                    }
                ],
                "reason": "fixture",
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "preview"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/preview_life.py",
            "--config",
            str(config_path),
            "--at",
            "2026-11-17T13:00:00+03:00",
            "--output",
            str(output),
            "--replay",
            str(path),
        ],
        cwd=Path(__file__).resolve().parents[1],
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=15,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "0 API calls" in result.stdout
    assert "23:00" in (output / "report.md").read_text(encoding="utf-8")
    assert not (tmp_path / "memory").exists()

    class Offline:
        def complete(self, messages):
            raise AssertionError("No network")

    bird = Mockingbird.from_config(config, model=Offline())
    assert bird.life is not None
    assert bird.life.user == "Nikita"
    assert bird.memory is None
    assert not bird.life.directory.exists()
