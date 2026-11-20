import os
import re
from pathlib import Path
from uuid import uuid4

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"
ENV_PATH = PROJECT_ROOT / ".env"


def save_business_connection_id(value: str, path: Path = ENV_PATH) -> bool:
    """Atomically update one dotenv field without copying secrets to logs."""
    if not value or any(char in value for char in ("\r", "\n", "\x00")):
        raise ValueError("Invalid Business Connection ID")
    original = path.read_bytes() if path.exists() else b""
    content = original.decode("utf-8-sig")
    newline = "\r\n" if "\r\n" in content else "\n"
    # Quote the opaque ID without changing any other dotenv fields.
    replacement = (
        "TELEGRAM_BUSINESS_CONNECTION_ID='"
        + value.replace("\\", "\\\\").replace("'", "\\'")
        + "'"
    )
    pattern = re.compile(
        r"^(?:export\s+)?TELEGRAM_BUSINESS_CONNECTION_ID\s*=.*$", re.MULTILINE
    )
    matches = list(pattern.finditer(content))
    if matches:
        updated = pattern.sub(
            lambda match: replacement + ("\r" if match.group().endswith("\r") else ""),
            content,
        )
    else:
        updated = (
            content
            + (newline if content and not content.endswith("\n") else "")
            + replacement
            + newline
        )
    if updated == content:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid4().hex + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as stream:
            stream.write(updated)
            stream.flush()
            os.fsync(stream.fileno())
        if (path.read_bytes() if path.exists() else b"") != original:
            raise RuntimeError(
                ".env changed while saving connection ID; retry discovery"
            )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return True


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

    paths = config.setdefault("paths", {})
    paths.setdefault("runtime", str(Path(paths["processed"]).parent / "runtime"))
    paths.setdefault("memory", str(Path(paths["runtime"]) / "memory"))
    for key in ("raw", "processed", "runtime", "memory"):
        value = Path(config["paths"][key])
        if not value.is_absolute():
            value = PROJECT_ROOT / value
        config["paths"][key] = value

    return config


if __name__ == "__main__":
    config = load_config()
    print("Configuration loaded successfully")
