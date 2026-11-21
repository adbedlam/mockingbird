from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from mockingbird.chat.chat import Chat
from mockingbird.chat.event_recorder.event_recorder import PipelineEvent
from mockingbird.mockingbird.draft import Draft
from mockingbird.mockingbird.mockingbird import Mockingbird
from mockingbird.scheduler.scheduler import Decision


def create_bird(
    config: dict, *, seed: int | None = None, fast_replies: bool = False
) -> Mockingbird:
    if fast_replies:
        config.setdefault("scheduler", {})["reply_mode"] = "immediate"
    return Mockingbird.from_config(config, seed)


def create_pipeline(
    config: dict,
    mockingbird: Mockingbird,
    directory: Path,
    on_event: Callable[[PipelineEvent], None],
    *,
    max_generations: int | None = None,
    delivery: Callable[[Decision, Draft, datetime, Path], dict | None] | None = None,
    state_sink: Callable[[], None] | None = None,
) -> Chat:
    settings = config["inference_model"]
    return Chat(
        mockingbird,
        directory,
        on_event,
        model_info={
            key: settings[key]
            for key in ("host", "model", "temperature", "json_mode")
            if key in settings
        },
        max_generations=max_generations,
        delivery=delivery,
        state_sink=state_sink,
    )
