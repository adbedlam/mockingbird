import argparse
import logging
import sys
from pathlib import Path

from mockingbird.config import load_config
from mockingbird.pipeline import Pipeline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--diary-every",
        type=int,
        help="Override the diary threshold, counted in incoming messages, for this run.",
    )
    parser.add_argument(
        "--preview-diary",
        action="store_true",
        help="Write this chat to a separate preview diary, not the main diary.",
    )
    parser.add_argument(
        "--preview-name",
        default="chat-preview",
        help="Folder name under paths.diary when --preview-diary is enabled.",
    )
    args = parser.parse_args()
    if args.diary_every is not None and args.diary_every < 1:
        parser.error("--diary-every must be at least 1")
    if (
        args.preview_name in {".", ".."}
        or Path(args.preview_name).name != args.preview_name
    ):
        parser.error("--preview-name must be a single folder name")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )
    for stream in (sys.stdin, sys.stdout):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="strict")
    config = load_config()
    diary_path_root = (
        config["paths"]["diary"] / args.preview_name if args.preview_diary else None
    )
    pipeline = Pipeline.from_config(
        config,
        diary_update_every_messages=args.diary_every,
        diary_path_root=diary_path_root,
        load_recent_context=True,
    )
    logging.info("Pipeline is ready. Type Ctrl+C or send EOF to stop.")

    while True:
        try:
            user_input = input("other> ")
        except (EOFError, KeyboardInterrupt):
            print()
            logging.info("Pipeline stopped")
            break

        response = pipeline.respond_to(user_input)
        print(f"persona> {response}")


if __name__ == "__main__":
    main()
