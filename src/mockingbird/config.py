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

    config.setdefault("llm", {})["api_key"] = os.getenv("LLM_API_KEY")

    for key, value in config.get("paths", {}).items():
        path = Path(value)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        config["paths"][key] = path
    return config


if __name__ == "__main__":
    print(load_config())
