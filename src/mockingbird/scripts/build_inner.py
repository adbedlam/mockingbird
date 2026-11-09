import argparse
import logging

from mockingbird.config import load_config
from mockingbird.conversation.history import History
from mockingbird.pipeline import InnerPromptPipeline
from mockingbird.preprocessing.factory import make_processor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--force",
        action="store_true",
        help="Discard the saved chunk checkpoint and rebuild every profile section.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Count all messages and profile chunks without calling a model.",
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
    processor_config = config["preprocessing"]
    processor = make_processor(processor_config["messenger"], None)
    full_history = processor.process(user, other, config["paths"]["raw"])
    participants = {user, other}
    messages = [
        message for message in full_history.messages if message.author in participants
    ]
    messages.sort(key=lambda message: message.timestamp)
    if not messages:
        raise ValueError("No messages from the configured participants were found.")

    history = History(user, other, messages)
    output_path = (
        config["paths"]["processed"]
        / f"{user}_{other}".replace(" ", "_")
        / "inner_prompt.txt"
    )
    chunk_size = config.get("profile_build", {}).get("chunk_size", 500)
    chunk_count = (len(messages) + chunk_size - 1) // chunk_size
    logging.info(
        "Building the persona profile from %d full-history messages", len(messages)
    )
    logging.info("This requires %d chunk calls and one consolidation call", chunk_count)
    if args.dry_run:
        logging.info("Dry run complete; no model was called")
        return

    builder = InnerPromptPipeline.from_config(config)
    builder.build(history, output_path, force=args.force)


if __name__ == "__main__":
    main()
