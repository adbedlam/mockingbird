import base64
import mimetypes
from pathlib import Path

from mockingbird.llm.client import LLM
from mockingbird.llm.message import Message, Role
from mockingbird.preprocessing.prompts.vision import VISION_PROMPT


class VisionModel:
    def __init__(
        self,
        host: str,
        model: str,
        api_key: str | None = None,
        temperature: float = 0.5,
    ):
        self.client = LLM(host, model, api_key, temperature)

    def describe(self, image_path: Path) -> str:
        mime_type = mimetypes.guess_type(image_path)[0]

        image_data = base64.b64encode(image_path.read_bytes()).decode("utf-8")

        content = [
            {
                "type": "image_url",
                "image_url": {"url": f"data:{mime_type};base64,{image_data}"},
            }
        ]

        return self.client.complete(
            [
                Message(role=Role.SYSTEM, content=VISION_PROMPT),
                Message(role=Role.USER, content=content),
            ]
        )


def main() -> None:
    from mockingbird.config import load_config

    config = load_config()["vision_model"]

    vision = VisionModel(
        config["host"],
        config["model"],
        config["api_key"],
        config["temperature"],
    )

    image_path = Path(__file__).with_name("image.png")
    print(vision.describe(image_path))


if __name__ == "__main__":
    main()
