"""Split a Telegram personal-chat export into per-participant transcripts.

Reads a `result.json` Telegram export, groups messages by sender, and writes
one plain-text transcript per participant, in a format readable by an LLM
(one message per line, timestamp + text, chronological order).
"""

import argparse
from collections import defaultdict
from pathlib import Path

from mockingbird.datacls import Message
from mockingbird.preprocessing.telegram.by_user_processor import UserTelegramProcessor


def format_transcript(messages: list[Message]) -> str:
    lines = [
        f"[{m.timestamp:%Y-%m-%d %H:%M}] {m.text}"
        for m in sorted(messages, key=lambda m: m.timestamp)
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="path to Telegram result.json export")
    parser.add_argument(
        "output_dir",
        type=Path,
        help="directory to write per-user .txt transcripts into",
    )
    args = parser.parse_args()

    messages = UserTelegramProcessor().process(args.input)

    by_user: dict[int, list[Message]] = defaultdict(list)
    for m in messages:
        by_user[m.user_id].append(m)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for user_id, user_messages in by_user.items():
        out_path = args.output_dir / f"messages_{user_id}.txt"
        out_path.write_text(format_transcript(user_messages), encoding="utf-8")
        print(f"user_id={user_id}: {len(user_messages)} messages -> {out_path}")


if __name__ == "__main__":
    main()
