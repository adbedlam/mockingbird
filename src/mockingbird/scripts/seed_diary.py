import argparse
import logging
from dataclasses import replace
from pathlib import Path

from mockingbird.config import load_config
from mockingbird.conversation.history import History
from mockingbird.pipeline import InnerPromptPipeline, Pipeline
from mockingbird.preprocessing.telegram.processor import TelegramProcessor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Count matching messages and show the selected date range without calling a model.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )
    config = load_config()
    user = config["user"]
    other = config["other"]
    threshold = config.get("diary", {}).get("update_every_messages", 100)

    history = TelegramProcessor().process(
        user,
        other,
        config["paths"]["raw"],
    )
    exact_messages = [
        message for message in history.messages if message.author in {user, other}
    ]
    participants = {user.strip().casefold(): user, other.strip().casefold(): other}
    messages = [
        replace(message, author=participants[message.author.strip().casefold()])
        for message in history.messages
        if message.author.strip().casefold() in participants
    ]
    logging.info(
        "Participant matches: exact=%d, normalized=%d",
        len(exact_messages),
        len(messages),
    )
    if args.dry_run:
        if len(messages) >= threshold:
            selected = messages[-threshold:]
            logging.info(
                "Would select %d messages from %s through %s; no model call made",
                len(selected),
                selected[0].timestamp.isoformat(),
                selected[-1].timestamp.isoformat(),
            )
        else:
            logging.info("Below threshold %d; no model call made", threshold)
        return
    if len(messages) < threshold:
        raise ValueError(
            "The Telegram export has fewer participant messages than the configured diary threshold."
        )

    selected_messages = messages[-threshold:]
    prompt_path: Path = (
        config["paths"]["processed"]
        / f"{user}_{other}".replace(" ", "_")
        / "inner_prompt.txt"
    )
    if not prompt_path.exists():
        builder = InnerPromptPipeline.from_config(config)
        builder.build(History(user, other, messages), prompt_path)

    pipeline = Pipeline.from_config(config)
    pipeline.feed(selected_messages)
    logging.info(
        "Submitted the latest %d participant message(s) to the diary", threshold
    )


if __name__ == "__main__":
    main()
