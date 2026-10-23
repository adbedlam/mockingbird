from mockingbird.preprocessing.telegram.processor import TelegramProcessor


def make_processor(messenger: str):
    if messenger == "telegram":
        return TelegramProcessor()

    raise SystemExit(f"Cannot pick a preprocessor for {messenger}")
