"""Inspect a simulated day and one reply in an isolated directory.

--prepare never calls a model. --generate makes at most two inference calls.
--replay exercises the clock using a saved plan, without a model or API key.
"""

import argparse
import json
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from mockingbird.config import CONFIG_PATH, PROJECT_ROOT, load_config
from mockingbird.mockingbird.mockingbird import Mockingbird
from mockingbird.persona.life import PersonaLife
from mockingbird.scheduler.profile import BehaviorProfile
from mockingbird.scheduler.scheduler import Scheduler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--at", help="Timezone-aware ISO date/time, defaults to now")
    parser.add_argument("--message", default="как дела что делал сегодня")
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--prepare", action="store_true")
    modes.add_argument("--generate", action="store_true")
    modes.add_argument(
        "--replay", type=Path, help="Replay a saved life plan without API calls"
    )
    args = parser.parse_args()
    config = load_config(args.config)
    zone = ZoneInfo(config.get("scheduler", {}).get("timezone", "Europe/Moscow"))
    now = datetime.fromisoformat(args.at) if args.at else datetime.now(zone)
    if now.tzinfo is None:
        parser.error(
            "--at needs a timezone offset, for example 2026-11-17T13:00:00+03:00"
        )
    now = now.astimezone(zone)
    output = args.output or PROJECT_ROOT / "data/previews/life" / datetime.now(
        zone
    ).strftime("%Y%m%d-%H%M%S-%f")
    if (output / "life" / "life.json").exists():
        parser.error(
            "This preview already has a plan; choose another --output directory"
        )

    # Offline modes cannot construct a network client accidentally.
    class Offline:
        def complete(self, messages):
            raise RuntimeError("Offline preview cannot call a model")

    bird = Mockingbird.from_config(
        config, model=None if args.generate else Offline(), memory_enabled=False
    )
    config_reply = bird.scheduler.config
    bird.scheduler = Scheduler(
        BehaviorProfile(bird.history, replace(config_reply, reply_mode="immediate")), 7
    )
    life = PersonaLife(output / "life", bird.history.user, bird.history.other, zone.key)
    bird.life = life
    messages = life.prepare(now, bird.inner_prompt.content)
    output.mkdir(parents=True, exist_ok=True)
    request_path = output / "life.request.json"
    request_path.write_text(
        json.dumps(
            {
                "at": now.isoformat(),
                "kind": "life_plan",
                "model": {
                    k: config["inference_model"][k]
                    for k in ("host", "model", "temperature")
                    if k in config["inference_model"]
                },
                "messages": messages,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    if args.prepare:
        print(f"Prepared locally; 0 API calls. Request: {request_path}")
        return
    raw = (
        bird.model.complete(messages)
        if args.generate
        else args.replay.read_text(encoding="utf-8")
    )
    (output / "life.response.txt").write_text(raw, encoding="utf-8")
    life.apply(raw, now.date().isoformat(), now)
    report = [
        "# Проверка вымышленного дня",
        "\nИсточник: "
        + (
            "настоящий ответ API"
            if args.generate
            else f"локальное воспроизведение {args.replay}"
        ),
        "\n## План на момент запуска\n\n" + life.view(now),
    ]
    if args.generate:
        bird.receive(args.message, now)
        due = bird.scheduler.reply_due
        assert due is not None
        decision = bird.scheduler.tick(due)
        reply_messages = bird.context(decision, due)
        reply_path = output / "reply.request.json"
        reply_path.write_text(
            json.dumps(
                {"at": due.isoformat(), "messages": reply_messages},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        raw_reply = bird.model.complete(reply_messages)
        (output / "reply.response.txt").write_text(raw_reply, encoding="utf-8")
        draft = bird.parse_draft(raw_reply, decision.kind)
        report.append(
            f"\n## Ответ\n\nВходящее: {args.message}\n\n{draft.text}\n\nПояснение: {draft.reason}"
        )
    for hour in (8, 13, 20, 23):
        at = now.replace(hour=hour, minute=0, second=0, microsecond=0)
        report.append(f"\n## Время {hour}:00\n\n" + life.view(at))
    life.advance(now.replace(hour=23, minute=59), checkpoint=True)
    report.append("\n## Итог дня (прокрутка часов)\n\n" + life.diary_view())
    (output / "report.md").write_text("\n".join(report), encoding="utf-8")
    print(life.view(now))
    print(
        f"Preview complete; {'2' if args.generate else '0'} API calls. Report: {output / 'report.md'}"
    )


if __name__ == "__main__":
    main()
