from .communication_style import CommunicationStyle
from .core_personality import CorePersonality


class Persona:
    core_personality: CorePersonality
    communication_style: CommunicationStyle

    def __init__(
        self, core_personality: CorePersonality, communication_style: CommunicationStyle
    ):
        self.core_personality = core_personality
        self.communication_style = communication_style

    def generate_system_prompt(self) -> str:
        messages = "\n".join(
            message.text for message in self.communication_style.sample_messages
        )
        return f"You are to imitate the behavior of a person based on these message examples:\n{messages}"
