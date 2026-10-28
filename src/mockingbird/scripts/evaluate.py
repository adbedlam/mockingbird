from collections.abc import Iterator
from pathlib import Path
from itertools import islice

from mockingbird.conversation.message import Message
from mockingbird.conversation.history import History
from mockingbird.preprocessing.factory import make_processor
from mockingbird.mockingbird import Mockingbird
from mockingbird.config import load_config

def main() -> None:
    config = load_config()

    filename = config["filename"]
    user = config["user"]
    other = config["other"]

    raw_file: Path = config["paths"]["raw"] / filename

    processor = make_processor(config["messenger"])

    history = processor.process(raw_file, user, other)
    
    message_config = config["message_selection"]
    message_start = message_config["evaluation_start"]
    message_end = message_start + message_config["evaluation_num"]

    history.messages = history.messages[message_start:message_end]

    window_size = message_config["evaluation_window"]
    sequences = find_sequences(history, user, other, window_size)
    
    processed_path: Path = config["paths"]["processed"]
    processed_path = processed_path / f"{user}_{other}".replace(" ", "_")
    processed_path.mkdir(parents=True, exist_ok=True)
    output_path: Path = processed_path / "evaluation.txt"

    mockingbird = Mockingbird()

    max_predictions = message_config["max_predictions"]

    with output_path.open("w", encoding="utf-8") as file:
        for sequence in islice(sequences, max_predictions):
            file.write("=" * 80 + "\n")

            mockingbird.feed(sequence[:-1])

            for message in sequence[:-1]:
                file.write(f"{message.author}: {message.text}\n")
            
            file.write("actual: ")
            file.write(f"{sequence[-1].text}\n")

            file.write("mockingbird: ")
            file.write(mockingbird.chirp() + "\n\n")

            mockingbird.kill()
    
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