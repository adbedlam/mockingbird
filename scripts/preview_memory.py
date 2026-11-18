"""Inspect memory deduplication and a completed task using synthetic evidence.

--prepare saves one exact request locally. --generate makes at most three
inference calls. No real profile, Telegram export, or production memory is sent.
"""

import argparse
import json
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from mockingbird.config import CONFIG_PATH, PROJECT_ROOT, load_config
from mockingbird.conversation.message import Message
from mockingbird.llm.client import LLM
from mockingbird.persona.diary.diary import Diary
from mockingbird.persona.diary.runtime import DiaryMemory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an interrupted preview; reuse saved valid responses without new API calls",
    )
    parser.add_argument("--at", help="Timezone-aware virtual starting time")
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--prepare", action="store_true")
    modes.add_argument("--generate", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    timezone = config.get("scheduler", {}).get("timezone", "Europe/Moscow")
    zone = ZoneInfo(timezone)
    start = datetime.fromisoformat(args.at) if args.at else datetime.now(zone)
    if start.tzinfo is None:
        parser.error("--at needs a timezone offset")
    start = start.astimezone(zone)
    output = args.output or PROJECT_ROOT / "data/previews/memory" / datetime.now(
        zone
    ).strftime("%Y%m%d-%H%M%S-%f")
    if output.exists() and not args.resume:
        parser.error("Choose a new output directory; old previews are preserved")
    if args.resume and not output.exists():
        parser.error("--resume needs an existing --output directory")
    if args.resume:
        previous = json.loads((output / "1.request.json").read_text(encoding="utf-8"))
        original_evidence = json.loads(previous["body"]["messages"][-1]["content"])[
            "incoming_evidence"
        ][0]
        original_start = datetime.fromisoformat(original_evidence["timestamp"])
        if args.at and start != original_start:
            parser.error("--resume must use the original virtual starting time")
        start = original_start
    memory = DiaryMemory(
        output / "memory",
        "Persona",
        "Partner",
        Diary(),
        timezone=timezone,
        update_every=1,
    )
    settings = config["inference_model"]
    model = None
    cases = [
        (0, "Вечером собираюсь доделать тесты проекта"),
        (30, "На сегодня всё ещё хочу закончить проектные тесты"),
        (180, "Тесты проекта уже закончил можно закрыть эту задачу"),
    ]
    report = [
        "# Проверка памяти",
        "Все реплики искусственные. Рабочая память, профиль и исходная переписка не используются. Время сообщений виртуальное.",
    ]
    report_path = output / "report.md"
    result_path = output / "results.json"
    if args.resume and report_path.exists():
        report = [report_path.read_text(encoding="utf-8")]
    results = (
        json.loads(result_path.read_text(encoding="utf-8"))
        if args.resume and result_path.exists()
        else []
    )
    calls, reused = 0, 0
    for index, (minutes, text) in enumerate(cases, 1):
        now = start + timedelta(minutes=minutes)
        message = Message("Partner", now, text=text)
        if memory.fingerprint(message) in memory.processed:
            continue
        memory.ingest([message])
        batch = memory.prepare(now)
        payload = {
            "method": "POST",
            "endpoint": settings["host"].rstrip("/") + "/chat/completions",
            "body": {
                "model": settings["model"],
                "temperature": settings.get("temperature", 0.7),
                "messages": batch.messages,
            },
        }
        if settings.get("json_mode", True):
            payload["body"]["response_format"] = {"type": "json_object"}
        request_path = output / f"{index}.request.json"
        response_path = output / f"{index}.response.txt"
        if request_path.exists() and not response_path.exists():
            previous_payload = json.loads(request_path.read_text(encoding="utf-8"))
            if previous_payload != payload:
                (output / f"{index}.previous.request.json").write_bytes(
                    request_path.read_bytes()
                )
                request_path.write_text(
                    json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
                )
        if not request_path.exists():
            request_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        if args.prepare:
            print(f"0 API calls. Synthetic request: {request_path}")
            return
        if args.resume and response_path.exists():
            raw = response_path.read_text(encoding="utf-8")
            reused += 1
        else:
            if model is None:
                model = LLM(
                    settings["host"],
                    settings["model"],
                    settings.get("api_key"),
                    settings.get("temperature", 0.7),
                    timeout=settings.get("timeout_seconds", 60),
                    json_mode=settings.get("json_mode", True),
                )
            raw = model.complete(batch.messages)
            calls += 1
            response_path.write_text(raw, encoding="utf-8")
        result = memory.apply(batch, raw, now)
        results.append(asdict(result))
        current = memory.combined().entries
        line = f"Шаг {index}: добавлено {result.added}, подтверждено {result.confirmed}, уточнено {result.updated}, спорно {result.disputed}, пропущено {result.skipped}; актуальных записей {len(current)}"
        print(line, flush=True)
        report.extend(
            [
                f"\n## {now.isoformat()}\n\n{text}\n\n{line}\n\nПояснение: {result.reason}",
                "\n".join(entry.to_string() for entry in current),
            ]
        )
        (output / "results.json").write_text(
            json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (output / "report.md").write_text("\n\n".join(report), encoding="utf-8")
    current = memory.combined().entries
    if (
        len(current) != 1
        or current[0].state != "completed"
        or current[0].subject != "partner"
    ):
        raise RuntimeError(
            f"Semantic check needs review; inspect {output / 'report.md'}"
        )
    print(
        f"New API calls: {calls}; saved responses reused: {reused}. One current completed memory. Report: {output / 'report.md'}"
    )


if __name__ == "__main__":
    main()
