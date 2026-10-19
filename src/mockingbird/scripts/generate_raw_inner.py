import argparse
from pathlib import Path

from mockingbird.paths import PROJECT_ROOT
from mockingbird.persona.inner import raw_prompt
from mockingbird.preprocessing.factory import make_processor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", type=Path)
    parser.add_argument("user")
    parser.add_argument("other")
    parser.add_argument(
        "--output-root", type=Path, default=PROJECT_ROOT / "data" / "processed"
    )
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    processor = make_processor(args.data_path)
    history = processor.process(args.user, args.other, args.data_path)

    if not history.users_messages():
        raise SystemExit(f"No messages found for {args.user!r} in {args.data_path}")

    prompt = raw_prompt(history, args.max_samples)

    out_dir = args.output_root / "inner" / args.user.replace(" ", "_")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "raw_inner.txt"
    out_file.write_text(prompt, encoding="utf-8")

    print(f"Wrote raw prompt to {out_file} ({len(prompt)} chars)")


if __name__ == "__main__":
    main()
