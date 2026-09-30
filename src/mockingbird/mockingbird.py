from mockingbird.models.model import Model
from mockingbird.persona.persona import Persona


class Mockingbird:
    def __init__(self, persona: Persona, model: Model):
        self.persona = persona
        self.model = model
        self.history = [{"role": "system", "content": persona.generate_system_prompt()}]

    def mock(self) -> None:
        while True:
            try:
                user_input = input("user> ").strip()
            except (EOFError, KeyboardInterrupt):
                break

            if not user_input:
                continue

            self.history.append({"role": "user", "content": user_input})

            try:
                response = self.model.generate(self.history)
            except Exception as exception:
                print(f"[error] {exception}")

                self.history.pop()

                continue

            self.history.append({"role": "assistant", "content": response.text})

            print(f"mockingbird> {response}")


if __name__ == "__main__":
    from mockingbird.models.qwen.model import Qwen

    qwen = Qwen(model="qwen3-4b-instruct-2507")

    from mockingbird.preprocessing.telegram.processor import TelegramProcessor

    processor = TelegramProcessor()

    from pathlib import Path

    from mockingbird.persona.core_personality import CorePersonality

    core_personality = CorePersonality(Path("data/processed/empty.txt"))

    from mockingbird.persona.communication_style import CommunicationStyle

    communication_style = CommunicationStyle(
        Path("data/raw/telegram/Nikita_Yaneev.json"), processor
    )

    from mockingbird.persona.persona import Persona

    persona = Persona(core_personality, communication_style)

    mockingbird = Mockingbird(persona, qwen)

    mockingbird.mock()
