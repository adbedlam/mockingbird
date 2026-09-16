from src.mockingbird.preprocessing.message import Message

class Prompt:
    def build(
            self,
            examples: list[Message],
            user_message: str
    ) -> list[dict[str, str]]:
        pass