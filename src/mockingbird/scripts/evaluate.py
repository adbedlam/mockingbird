from collections.abc import Iterator
from itertools import islice
from pathlib import Path

from mockingbird.config import load_config
from mockingbird.conversation.history import History
from mockingbird.conversation.message import Message
from mockingbird.llm.client import LLM
from mockingbird.pipeline import Pipeline
from mockingbird.preprocessing.factory import make_processor
from mockingbird.preprocessing.vision import VisionModel


def main() -> None:
    config = load_config()

    filename = config["filename"]
    user = config["user"]
    other = config["other"]

    data_dir: Path = config["paths"]["raw"] / filename

    vision_config = config["vision_model"]
    vision_model = VisionModel(
        vision_config["host"],
        vision_config["model"],
        vision_config["api_key"],
        vision_config["temperature"],
    )
    evaluation_config = config["evaluation"]
    processor = make_processor(evaluation_config["messenger"], vision_model)

    history = processor.process(
        user,
        other,
        data_dir,
        evaluation_config["message_start"],
        evaluation_config["message_end"],
    )

    sequences = find_sequences(history, user, other, evaluation_config["window_size"])

    inference_config = config["inference_model"]
    model = LLM(
        inference_config["host"],
        inference_config["model"],
        inference_config["api_key"],
        inference_config["temperature"],
    )

    processed_dir = config["paths"]["processed"] / f"{user}_{other}".replace(" ", "_")
    processed_dir.mkdir(parents=True, exist_ok=True)
    output_path: Path = processed_dir / "evaluation.txt"

    prompt_path = (
        config["paths"]["processed"]
        / f"{user}_{other}".replace(" ", "_")
        / "inner_prompt.txt"
    )
    pipeline = Pipeline(
        user,
        other,
        model,
        prompt_path.read_text(encoding="utf-8"),
    )

    max_predictions = evaluation_config["max_predictions"]

    with output_path.open("w", encoding="utf-8") as file:
        for sequence in islice(sequences, max_predictions):
            file.write("=" * 80 + "\n")

            pipeline.feed(sequence[:-1])

            for message in sequence[:-1]:
                file.write(f"{message.author}: {message.text}\n")

            file.write("actual: ")
            file.write(f"{sequence[-1].text}\n")

            file.write("mockingbird: ")
            file.write(pipeline.respond() + "\n\n")

            pipeline.starve()


def find_sequences(
    history: History,
    user: str,
    other: str,
    window_size: int,
) -> Iterator[list[Message]]:
    messages = history.messages
    start = 0

    while start <= len(messages) - window_size:
        sequence = messages[start : start + window_size]
        authors = {message.author for message in sequence}

        if authors <= {user, other} and sequence[-1].author == user:
            yield sequence
            start += window_size
        else:
            start += 1


if __name__ == "__main__":
    main()
