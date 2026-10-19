from pathlib import Path

from mockingbird.preprocessing.telegram.processor import TelegramProcessor


def make_processor(data_path: Path):
    parts = set(data_path.resolve().parts)

    if "telegram" in parts:
        return TelegramProcessor()

    raise SystemExit(f"Cannot pick a preprocessor for {data_path}")
