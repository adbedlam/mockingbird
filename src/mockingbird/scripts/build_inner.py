from pathlib import Path

from mockingbird.config import load_config
from mockingbird.conversation.history import History
from mockingbird.llm.client import LLM
from mockingbird.persona.inner import build_inner_prompt
from mockingbird.preprocessing.factory import make_processor
from mockingbird.preprocessing.vision import VisionModel


def main() -> None:
    print("Loading config...")
    config = load_config()

    user = config["user"]
    other = config["other"]

    print(
        f"Building inner and local expression for {user}'s conversation with {other}..."
    )

    print("Creating a vision model...")
    vision_config = config["vision_model"]
    vision_model = VisionModel(
        vision_config["host"],
        vision_config["model"],
        vision_config["api_key"],
        vision_config["temperature"],
    )

    preprocessor_config = config["preprocessing"]
    print("Creating a processor")
    processor = make_processor(preprocessor_config["messenger"], vision_model)

    print("Processing the conversation...")

    data_dir = config["paths"]["raw"]
    history: History = processor.process(
        user,
        other,
        data_dir,
        preprocessor_config["message_start"],
        preprocessor_config["message_end"],
    )

    print("Creating a model...")
    building_config = config["building_model"]
    building_model = LLM(
        building_config["host"],
        building_config["model"],
        building_config["api_key"],
        building_config["temperature"],
    )

    print("Building the inner prompt...")
    inner = build_inner_prompt(user, other, history, building_model)

    processed_path: Path = config["paths"]["processed"]
    processed_path = processed_path / f"{user}_{other}".replace(" ", "_")
    processed_path.mkdir(parents=True, exist_ok=True)

    inner_prompt_file: Path = processed_path / "inner_prompt.txt"
    inner_prompt_file.write_text(inner, encoding="utf-8")
    print(f"Saved the inner prompt to {inner_prompt_file}")


if __name__ == "__main__":
    main()
