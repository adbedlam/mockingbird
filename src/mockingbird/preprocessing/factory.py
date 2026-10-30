from mockingbird.preprocessing.processor import Processor
from mockingbird.preprocessing.telegram.processor import TelegramProcessor
from mockingbird.preprocessing.vision import VisionModel


def make_processor(
    messenger: str, vision_model: VisionModel | None = None
) -> Processor:
    if messenger == "telegram":
        return TelegramProcessor(vision_model)

    raise SystemExit(f"Cannot pick a preprocessor for {messenger}")
