"""Accelerated scheduler rehearsal; --generate calls the configured LLM.

Nothing is sent to a messenger. All state is isolated in memory, and reports
are written under data/runtime/previews (or --output), never over the main diary.
"""

from __future__ import annotations

import argparse
import json
from datetime import date, datetime, time, timedelta
from pathlib import Path

from mockingbird.config import CONFIG_PATH, load_config
from mockingbird.llm.message import Message
from mockingbird.mockingbird.draft import Draft
from mockingbird.mockingbird.mockingbird import Mockingbird
from mockingbird.scheduler.scheduler import Action, MessageKind


class NoModel:
    def complete(self, messages: list[Message]) -> str:
        raise AssertionError("Dry-run must not call an LLM")


def write_report(path: Path, metadata: dict, events: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.with_suffix(".json").write_text(
        json.dumps(
            {"metadata": metadata, "events": events}, ensure_ascii=False, indent=2
        ),
        encoding="utf-8",
    )
    lines = [
        "# Предпросмотр scheduler",
        "",
        f"Период: {metadata['start']} — {metadata['end']}. Часовой пояс: {metadata['timezone']}.",
        f"Режим: {metadata['mode']}. Seed: {metadata['seed']}.",
        "Ничего не отправлено в Telegram. Основной дневник и состояние чата не менялись.",
        "Входящие в этом сценарии — искусственные ответы через заданную паузу после инициативы.",
        "В режиме без модели SEND означает разрешение scheduler; текст не генерируется.",
        "",
        f"Основа: {metadata['messages']} сообщений персоны, {metadata['observed_days']} календарных дней, {metadata['observed_initiations']} предполагаемых инициатив.",
        "Начало разговора определяется по паузе в переписке; это эвристика, не подтверждённое намерение человека.",
        "",
    ]
    for event in events:
        lines.extend([f"## {event['at']} — {event['action']}", "", event["reason"], ""])
        if event.get("text"):
            lines.extend(["> " + line for line in event["text"].splitlines()] + [""])
        if event.get("model_reason"):
            lines.extend(["Решение модели: " + event["model_reason"], ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument(
        "--date", type=date.fromisoformat, help="First virtual day, YYYY-MM-DD"
    )
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--seed", type=int, default=7)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--generate",
        action="store_true",
        help="Generate text using the configured inference model",
    )
    mode.add_argument(
        "--prepare-request",
        action="store_true",
        help="Write the exact first model request locally without sending it",
    )
    parser.add_argument("--max-generations", type=int, default=10)
    parser.add_argument(
        "--reply-after",
        type=int,
        default=900,
        help="Seconds until a synthetic incoming reply",
    )
    parser.add_argument(
        "--no-incoming",
        action="store_true",
        help="Leave initiatives unanswered to check suppression",
    )
    parser.add_argument(
        "--output",
        type=Path,
    )
    args = parser.parse_args()
    if not 1 <= args.days <= 365 or args.reply_after < 1 or args.max_generations < 1:
        parser.error(
            "days must be 1..365, reply-after and max-generations must be positive"
        )
    config = load_config(args.config)
    if args.output is None:
        args.output = Path(config["paths"]["runtime"]) / "previews/scheduler-preview.md"
    mockingbird = Mockingbird.from_config(
        config, args.seed, None if args.generate else NoModel(), memory_enabled=False
    )
    profile = mockingbird.scheduler.profile
    start = args.date or datetime.now(profile.zone).date()
    now = datetime.combine(start, time(), profile.zone)
    end = now + timedelta(days=args.days)
    metadata = {
        "start": start.isoformat(),
        "end": end.date().isoformat(),
        "timezone": profile.config["timezone"],
        "mode": (
            "Настоящая генерация текста LLM"
            if args.generate
            else (
                "Локальная подготовка запроса без отправки"
                if args.prepare_request
                else "Без вызова модели, только расписание"
            )
        ),
        "seed": args.seed,
        "messages": profile.message_count,
        "observed_days": profile.observed_days,
        "observed_initiations": profile.observed_initiations,
    }
    events: list[dict] = []
    incoming: list[datetime] = []
    shown_day = None
    generations = 0

    def log(action: str, reason: str, text: str = "", model_reason: str = "") -> None:
        event = {
            "at": now.isoformat(),
            "action": action,
            "reason": reason,
            "text": text,
            "model_reason": model_reason,
        }
        events.append(event)
        print(f"[{now.isoformat()}] {action}: {reason}")
        if text:
            print(text)
        if model_reason:
            print("Модель: " + model_reason)
        write_report(args.output, metadata, events)

    while now < end:
        while incoming and incoming[0] <= now:
            incoming.pop(0)
            text = "Привет! Что хотел обсудить?"
            mockingbird.receive(text, now)
            log("INCOMING (сценарий)", "Искусственный ответ собеседника", text)
            log(
                "REPLY_PLAN",
                f"Ответ запланирован на {mockingbird.scheduler.reply_due}; задержка взята из истории и перенесена в активное время при необходимости.",
            )
        decision = mockingbird.scheduler.tick(now)
        if shown_day != now.date():
            shown_day = now.date()
            times = (
                ", ".join(t.strftime("%H:%M:%S") for t in mockingbird.scheduler.planned)
                or "инициативы не запланированы"
            )
            log(
                "PLAN",
                f"План на день: {times}. Лимит: {profile.config['max_initiations_per_day']}.",
            )
        if decision.action == Action.SKIP:
            log("SKIP", decision.reason)
            continue
        if decision.action == Action.SEND:
            if args.prepare_request:
                settings = config["inference_model"]
                request_path = args.output.with_suffix(".request.json")
                request_path.parent.mkdir(parents=True, exist_ok=True)
                request_path.write_text(
                    json.dumps(
                        {
                            "at": now.isoformat(),
                            "kind": decision.kind,
                            "method": "POST",
                            "endpoint": settings["host"].rstrip("/")
                            + "/chat/completions",
                            "body": {
                                "model": settings["model"],
                                "temperature": settings.get("temperature", 0.7),
                                "messages": mockingbird.context_builder.build(
                                    decision, now
                                ),
                            },
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                log(
                    "REQUEST_PREPARED",
                    f"Точный первый запрос записан локально: {request_path}. Ключ не включён, запрос не отправлен.",
                )
                break
            if args.generate:
                if generations >= args.max_generations:
                    log("STOP", "Достигнут заданный лимит запросов к модели")
                    break
                generations += 1
                try:
                    draft = mockingbird.compose(decision, now)
                except Exception as exc:
                    log(
                        "ERROR",
                        f"Генерация не завершена: {type(exc).__name__}. Состояние не подтверждено как отправленное.",
                    )
                    raise
            else:
                draft = Draft(
                    True, "", "Модель не вызывалась; проверяем разрешение на отправку"
                )
            if args.generate:
                mockingbird.commit(decision, draft, now)
            else:
                # A timing acknowledgement is not a generated chat message.
                mockingbird.scheduler.mark_sent(decision, now)
            log(
                f"SEND {decision.kind}" if draft.send else "MODEL_SKIP",
                decision.reason,
                draft.text,
                draft.reason,
            )
            if (
                draft.send
                and decision.kind == MessageKind.INITIATIVE
                and not args.no_incoming
            ):
                incoming.append(now + timedelta(seconds=args.reply_after))
                incoming.sort()
            continue
        midnight = datetime.combine(
            now.date() + timedelta(days=1), time(), profile.zone
        )
        candidates = [midnight, end]
        if decision.due_at is not None and decision.due_at > now:
            candidates.append(decision.due_at)
        if incoming:
            candidates.append(incoming[0])
        now = min(candidates)
    write_report(args.output, metadata, events)
    print(f"Отчёт: {args.output}\nJSON: {args.output.with_suffix('.json')}")


if __name__ == "__main__":
    main()
