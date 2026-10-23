from pathlib import Path

import pandas as pd

from mockingbird.analytics.analyze import analyze
from mockingbird.analytics.annotate import annotate
from mockingbird.analytics.emoji import most_common_emojis
from mockingbird.analytics.lexicon import top_ngrams
from mockingbird.config import load_config
from mockingbird.llm.client import LLM
from mockingbird.persona.inner import (
    build_emoji_prompt,
    build_lexicon_prompt,
    build_personality_prompt,
)
from mockingbird.preprocessing.factory import make_processor


def main() -> None:
    print("Loading config...")
    config = load_config()

    filename = config["filename"]
    user = config["user"]
    other = config["other"]
    print(
        f"Building inner and local expression for {user}'s conversation with {other}..."
    )

    raw_path: Path = config["paths"]["raw"]
    raw_file: Path = raw_path / filename
    print(f"File path: {raw_file}")
    if not raw_file.is_file():
        raise SystemExit(f"input file not found: {raw_file}")
    processed_path: Path = config["paths"]["processed"]
    processed_path = processed_path / f"{user}_{other}".replace(" ", "_")
    processed_path.mkdir(parents=True, exist_ok=True)

    analytics_config = config["analytics"]

    # Read conversation history
    print("Reading conversation history...")
    processor = make_processor(config["messenger"])

    history = processor.process(raw_file, user, other)
    print(f"Reading the conversation yielded {len(history.messages)} messages")

    history.messages = history.messages[0 : analytics_config["max_messages"]]
    if not history.users_messages():
        raise SystemExit(f"No messages found for {user!r} in {raw_file.absolute()}")

    # Run the LLM independent analysis
    print(f"Running the LLM independent analysis")
    df = pd.DataFrame({"message": history.messages})
    df = pd.concat(
        [
            df,
            analyze(history, user),
        ],
        axis=1,
    )

    ngram_config = analytics_config["ngrams"]
    ngrams: list[list[str]] = [
        top_ngrams(history, n, ngram_config["top_k"])
        for n in range(1, ngram_config["max_n"] + 1)
    ]
    emojis: list[str] = most_common_emojis(history)
    print(f"Running the analysis produced a {df.shape[0]}-row DataFrame")
    # Create the model
    print("Creating a model...")
    llm_config = config["llm"]
    model = LLM(llm_config["host"], llm_config["model"], llm_config["api_key"])

    # Build inner prompt
    print("Building the inner prompt...")
    building_temperature = llm_config["building_temperature"]
    personality = build_personality_prompt(
        history, model, temperature=building_temperature
    )
    lexicon = build_lexicon_prompt(ngrams, model, temperature=building_temperature)
    emoji = build_emoji_prompt(emojis)
    inner = "\n\n".join(
        string.rstrip() for string in (personality, lexicon, emoji) if string
    )

    # Save inner prompt
    inner_prompt_file: Path = processed_path / "inner_prompt.txt"
    inner_prompt_file.write_text(inner, encoding="utf-8")
    print(f"Saved the inner prompt to {inner_prompt_file}")

    # Run the LLM dependent analytics
    print("Running the LLM dependent analytics...")
    annotations = annotate(history, model)
    df = pd.concat([df, annotations], axis=1)

    # Save analytics to train the local expression model
    analytics_file: Path = processed_path / "analytics.csv"
    df.to_csv(analytics_file, index=False, encoding="utf-8")
    print(f"Final DataFrame: {df.shape[0]} rows × {df.shape[1]} cols")
    print(f"Saved the analytics to {analytics_file}")

    # TODO: Train the local expression model


if __name__ == "__main__":
    main()
