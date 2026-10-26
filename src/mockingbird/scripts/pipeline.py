from pathlib import Path


from mockingbird.config import load_config
from mockingbird.llm.client import LLM
from mockingbird.persona.inner import build_inner_prompt
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

    print("Creating a model...")
    llm_config = config["llm"]
    model = LLM(llm_config["host"], llm_config["model"], llm_config["api_key"])

    # Build inner prompt
    print("Building the inner prompt...")
    building_temperature = llm_config["building_temperature"]
    inner = build_inner_prompt(
        history, model, temperature=building_temperature
    )

    # Save inner prompt
    inner_prompt_file: Path = processed_path / "inner_prompt.txt"
    inner_prompt_file.write_text(inner, encoding="utf-8")
    print(f"Saved the inner prompt to {inner_prompt_file}")


if __name__ == "__main__":
    main()
