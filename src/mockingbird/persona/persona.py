import numpy as np

from mockingbird.analysis.stats import (
    avg_message_len,
    emoji_frequency,
    message_bursts,
    most_common_emojis,
    response_latencies,
)
from mockingbird.datacls import (
    BehavioralPatterns,
    CommunicationStyle,
    Conversation,
    EmojiUse,
    Lexicon,
    MessagingPatterns,
    Username,
)


class Persona:
    def __init__(self, conversations: list[Conversation]):
        self.communication_styles: dict[Username, CommunicationStyle] = {}

        for conversation in conversations:
            messages = conversation.messages

            if not messages:
                continue

            persona_messages = [
                message for message in messages if not message.from_other
            ]

            bursts = message_bursts(persona_messages)
            latencies = [
                latency.total_seconds()
                for latency in response_latencies(persona_messages)
            ]

            messaging_patterns = MessagingPatterns(
                float(np.max(bursts)),
                float(np.mean(bursts)),
                np.median(latencies),
                np.mean(latencies),
            )

            emoji_use = EmojiUse(
                emoji_frequency(persona_messages), most_common_emojis(persona_messages)
            )

            lexicon = Lexicon([], [])

            self.communication_styles[conversation.user] = CommunicationStyle(
                avg_message_len(persona_messages),
                emoji_use,
                messaging_patterns,
                lexicon,
                [],  # TODO
            )

        self.behavioral_patterns = BehavioralPatterns()

    def generate_system_prompt(self, username: str) -> str:
        # TODO
        return f"act like {username}"
