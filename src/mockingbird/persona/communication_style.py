import random
from pathlib import Path

import numpy as np

from mockingbird.datacls import Conversation, Message
from mockingbird.preprocessing.processor import Processor

from .stats import (
    avg_message_len,
    emoji_frequency,
    message_bursts,
    most_common_emojis,
    response_latencies,
)


class MessageCharacteristics:
    avg_message_len: float

    def __init__(self, conversation: Conversation):
        messages = conversation.messages

        persona_messages = [message for message in messages if not message.from_other]

        self.avg_message_len = avg_message_len(persona_messages)


class MessagingPatterns:
    max_burst_len: float
    mean_burst_len: float
    median_response_latency: float
    mean_response_latency: float

    def __init__(self, conversation: Conversation):
        messages = conversation.messages

        persona_messages = [message for message in messages if not message.from_other]

        bursts = message_bursts(persona_messages)
        latencies = [
            latency.total_seconds() for latency in response_latencies(messages)
        ]

        self.max_burst_len = float(np.max(bursts))
        self.mean_burst_len = float(np.mean(bursts))
        self.median_response_latency = np.median(latencies)
        self.mean_response_latency = np.mean(latencies)


class EmojiUse:
    emoji_frequency: float
    most_common_emojis: list[tuple[str, int]]

    def __init__(self, conversation: Conversation):
        messages = conversation.messages

        persona_messages = [message for message in messages if not message.from_other]

        self.emoji_frequency = emoji_frequency(persona_messages)
        self.most_common_emojis = most_common_emojis(persona_messages)


class Lexicon:
    distinctive_words: list[tuple[str, float]]
    distinctive_phrases: list[tuple[str, float]]

    def __init__(self, conversation: Conversation):
        pass


class CommunicationStyle:
    message_characteristics: MessageCharacteristics
    emoji_use: EmojiUse
    messaging_patterns: MessagingPatterns
    lexicon: Lexicon
    sample_messages: list[Message]

    def __init__(self, conversation_path: Path, processor: Processor):
        conversation = processor.process(conversation_path)

        self.message_characteristics = MessageCharacteristics(conversation)
        self.emoji_use = EmojiUse(conversation)
        self.messaging_patterns = MessagingPatterns(conversation)
        self.lexicon = Lexicon(conversation)
        self.sample_messages = self._get_sample_messages(conversation)

    def _get_sample_messages(
        self, conversation: Conversation, num_samples: int = 350
    ) -> list[Message]:
        return random.sample(
            [message for message in conversation.messages if not message.from_other],
            num_samples,
        )
