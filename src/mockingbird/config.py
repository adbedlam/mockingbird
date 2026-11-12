import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"
ENV_PATH = PROJECT_ROOT / ".env"


def load_config(path: Path = CONFIG_PATH) -> dict:
    load_dotenv(ENV_PATH)

    config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    config.setdefault("building_model", {})["api_key"] = os.getenv(
        "BUILDING_MODEL_API_KEY"
    )
    config.setdefault("inference_model", {})["api_key"] = os.getenv(
        "INFERENCE_MODEL_API_KEY"
    )
    config.setdefault("vision_model", {})["api_key"] = os.getenv("VISION_MODEL_API_KEY")

    for key in ("raw", "processed"):
        value = Path(config["paths"][key])
        if not value.is_absolute():
            value = PROJECT_ROOT / value
        config["paths"][key] = value

    return config


if __name__ == "__main__":
    print(load_config())
