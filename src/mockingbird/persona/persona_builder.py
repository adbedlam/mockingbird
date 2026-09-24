import numpy as np

from mockingbird.analysis.stats import (
    avg_message_len,
    message_bursts,
    emoji_frequency,
    response_latencies,
    most_common_emojis,
)
from mockingbird.datacls import (
    BehavioralPatterns,
    CommunicationStyle,
    Conversation,
    EmojiUse,
    Lexicon,
    MessagingPatterns,
    Persona,
)


def build_persona(conversations: list[Conversation]) -> Persona:
    persona = Persona()

    for conversation in conversations:
        messages = conversation.messages

        if not messages:
            continue

        persona_messages = [message for message in messages if not message.from_other]

        bursts = message_bursts(persona_messages)
        latencies = response_latencies(persona_messages)

        messaging_patterns = MessagingPatterns(
            np.max(bursts),
            np.mean(bursts),
            np.median(latencies),
            np.mean(latencies),
        )

        emoji_use = EmojiUse(
            emoji_frequency(persona_messages), most_common_emojis(persona_messages)
        )

        lexicon = Lexicon([], [])  # idk about that honestly, fix pls

        persona.communication_styles[conversation.user] = CommunicationStyle(
            avg_message_len(persona_messages),
            emoji_use,
            messaging_patterns,
            lexicon,
            [],  # TODO
        )

    persona.behavioral_patterns = BehavioralPatterns([])  # TODO
