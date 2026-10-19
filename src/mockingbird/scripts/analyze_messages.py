import argparse
import random
from pathlib import Path

import pandas as pd

from mockingbird.analytics import emoji, length
from mockingbird.analytics.emotion import recognize_emotion
from mockingbird.conversation.message import Message
from mockingbird.llm.client import LLM
from mockingbird.paths import PROJECT_ROOT
from mockingbird.preprocessing.factory import make_processor


def _build_df(messages: list[Message], model: LLM) -> pd.DataFrame:
    rows = []
    for message in messages:
        emotion = recognize_emotion(message, model)
        rows.append(
            {
                "timestamp": message.timestamp,
                "text": message.text,
                "length": length.length(message),
                "emoji_count": emoji.message_emoji_count(message),
                "valence": emotion.valence,
                "arousal": emotion.arousal,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute per-message analytics from a chat export."
    )
    parser.add_argument("data_path", type=Path)
    parser.add_argument("user")
    parser.add_argument("other")
    parser.add_argument("--host", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key", default=None)
    parser.add_argument(
        "--output-root", type=Path, default=PROJECT_ROOT / "data" / "processed"
    )
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    processor = make_processor(args.data_path)
    history = processor.process(args.user, args.other, args.data_path)

    messages = history.users_messages()
    if not messages:
        raise SystemExit(f"No messages found for {args.user!r} in {args.data_path}")

    if args.max_samples and len(messages) > args.max_samples:
        messages = random.sample(messages, args.max_samples)
        messages.sort(key=lambda m: m.timestamp)

    model = LLM(args.host, args.model, args.api_key)

    df = _build_df(messages, model)

    out_dir: Path = args.output_root / "analytics"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{args.user.replace(' ', '_')}_messages.csv"
    df.to_csv(out_file, index=False, encoding="utf-8")

    print(f"Wrote {len(df)} rows for {args.user} to {out_file} with {args.model}")


if __name__ == "__main__":
    main()
