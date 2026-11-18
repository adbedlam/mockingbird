"""Demonstrate memory retrieval on synthetic records, with zero API calls."""

import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from mockingbird.config import PROJECT_ROOT
from mockingbird.persona.diary.entry import Entry
from mockingbird.persona.diary.retrieval import MemorySearchConfig, select_memories


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", default="что с интеграцией телеграма")
    parser.add_argument("--limit", type=int, default=6)
    parser.add_argument("--disable-search", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data/previews/memory-search/report.md",
    )
    args = parser.parse_args()
    now = datetime(2026, 11, 18, 13, tzinfo=ZoneInfo("Europe/Moscow"))
    entries = [
        Entry(
            now - timedelta(days=901),
            "Планировали интеграцию телеграма",
            memory_id="telegram",
            revision=0,
            state="planned",
        ),
        Entry(
            now - timedelta(days=900),
            "Партнёр закончил интеграцию телеграма",
            memory_id="telegram",
            revision=1,
            state="completed",
            subject="partner",
            topic="telegram.integration",
        ),
        *[
            Entry(
                now - timedelta(days=50 - index),
                f"Обсудили маршрут поездки номер {index}",
                memory_id=f"trip{index}",
            )
            for index in range(45)
        ],
    ]
    selection = select_memories(
        entries,
        args.query,
        now,
        timezone="Europe/Moscow",
        limit=args.limit,
        config=MemorySearchConfig(enabled=not args.disable_search),
    )
    report = [
        "# Локальный поиск воспоминаний",
        "Все записи искусственные. Вызовов API: 0. Рабочая память и профиль не используются.",
        f"Запрос: {args.query}",
        f"Проверено актуальных записей: {selection.considered}; выбрано: {len(selection.hits)}; бюджет: {selection.chars}/{selection.max_chars} символов.",
    ]
    for hit in selection.hits:
        report.extend(
            [
                f"\n## {hit.entry.memory_id} | {hit.reason} | score={hit.score:.3f}",
                hit.entry.to_string(),
                "Историческая запись: " + ("да" if hit.historical else "нет"),
                "Совпавшие основы слов: " + ", ".join(hit.matched_terms),
            ]
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n\n".join(report), encoding="utf-8")
    args.output.with_suffix(".json").write_text(
        json.dumps(selection.report(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"0 API calls. Selected {len(selection.hits)} of {selection.considered} current memories. Report: {args.output}"
    )


if __name__ == "__main__":
    main()
