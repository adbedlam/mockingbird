"""Inspect an agreed plan change and a follow-up in an isolated pipeline.

--prepare makes zero API calls. --generate uses at most two reply requests,
reusing a saved day plan. Input times are virtual; provider calls are real.
"""

import argparse
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from mockingbird.chat.pipeline import ChatPipeline
from mockingbird.config import CONFIG_PATH, load_config
from mockingbird.interfaces.terminal.interface import render
from mockingbird.mockingbird.mockingbird import Mockingbird
from mockingbird.persona.life import PersonaLife


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument(
        "--plan",
        type=Path,
        required=True,
        help="Existing life.json to copy; the original is never rewritten",
    )
    parser.add_argument("--at", help="Timezone-aware virtual start, defaults to now")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--message", default="давай вместо фильма вечером с 20 до 21:30 потестим проект"
    )
    parser.add_argument("--followup", default="а что у тебя вечером по плану")
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--prepare", action="store_true")
    modes.add_argument("--generate", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    zone = ZoneInfo(config.get("scheduler", {}).get("timezone", "Europe/Moscow"))
    now = datetime.fromisoformat(args.at) if args.at else datetime.now(zone)
    if now.tzinfo is None:
        parser.error("--at requires a timezone offset")
    now = now.astimezone(zone)
    output = args.output or Path(
        config["paths"]["runtime"]
    ) / "previews/life-changes" / datetime.now(zone).strftime("%Y%m%d-%H%M%S-%f")
    if output.exists():
        parser.error("Choose a new output directory; existing previews are preserved")
    config.setdefault("scheduler", {}).update(
        reply_mode="immediate", max_initiations_per_day=0
    )

    class Offline:
        def complete(self, messages):
            raise RuntimeError("Offline preparation cannot call a model")

    bird = Mockingbird.from_config(
        config, model=None if args.generate else Offline(), memory_enabled=False
    )
    life_dir = output / "life"
    life_dir.mkdir(parents=True)
    (life_dir / "life.json").write_bytes(args.plan.read_bytes())
    life = PersonaLife(
        life_dir,
        bird.history.user,
        bird.history.other,
        zone.key,
        life_prompt=bird.life_prompt.content,
    )
    if now.date().isoformat() not in life.days:
        parser.error(
            "The copied plan has no day for --at; use its date or another plan"
        )
    bird.life = life
    if args.prepare:
        bird.receive(args.message, now)
        due = bird.scheduler.reply_due
        assert due is not None
        messages = bird.context_builder.build(bird.scheduler.tick(due), due)
        settings = config["inference_model"]
        request = {
            "method": "POST",
            "endpoint": settings["host"].rstrip("/") + "/chat/completions",
            "body": {
                "model": settings["model"],
                "temperature": settings.get("temperature", 0.7),
                "messages": messages,
            },
        }
        if settings.get("json_mode", True):
            request["body"]["response_format"] = {"type": "json_object"}
        path = output / "change.request.json"
        path.write_text(
            json.dumps(request, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"0 API calls. Prepared request: {path}")
        return
    events = []

    def on_event(event):
        events.append(event)
        render(event)

    settings = config["inference_model"]
    pipeline = ChatPipeline(
        bird,
        output / "runtime",
        on_event,
        clock=lambda: now,
        max_generations=2,
        model_info={
            key: settings[key]
            for key in ("host", "model", "temperature", "json_mode")
            if key in settings
        },
    )
    try:
        pipeline.start()
        for text in (args.message, args.followup):
            outputs = sum(e.stage == "OUTPUT" for e in events)
            before = len(events)
            pipeline.receive(text)
            due = bird.scheduler.reply_due
            assert due is not None
            now = due
            deadline = time.monotonic() + settings.get("timeout_seconds", 60) + 15
            while sum(e.stage == "OUTPUT" for e in events) == outputs:
                pipeline.step()
                if any(
                    e.stage in ("ERROR", "LIFE_ERROR", "PAUSED")
                    for e in events[before:]
                ):
                    raise RuntimeError(
                        f"Reply was not accepted; inspect {output / 'runtime/requests'}"
                    )
                if time.monotonic() >= deadline:
                    raise TimeoutError("Preview timed out")
                time.sleep(0.05)
            now += timedelta(seconds=30)
        report = [
            "# Проверка изменения плана",
            "Настоящие API-запросы; часы диалога виртуальные. Исходный план и рабочая память не менялись.",
            "\n## Ответы\n",
        ]
        for event in events:
            if event.stage in ("INPUT", "OUTPUT", "LIFE_CHANGED"):
                report.append(
                    f"\n### {event.at} | {event.stage}\n\n{event.message}\n\n{event.data.get('text', '')}\n\n{event.data.get('reason', '')}"
                )
        report.append("\n## План после диалога\n\n" + life.view(now))
        now = now.replace(hour=23, minute=59, second=0, microsecond=0)
        pipeline.close()
        report.append("\n## Дневник после прокрутки часов\n\n" + life.diary_view())
        (output / "report.md").write_text("\n".join(report), encoding="utf-8")
        print(f"{pipeline.generations} API calls. Report: {output / 'report.md'}")
    finally:
        pipeline.close()


if __name__ == "__main__":
    main()
