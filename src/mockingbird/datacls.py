from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Message:
    text: str
    timestamp: datetime
    from_other: bool


@dataclass
class Conversation:
    chat_id: str
    user: str
    messages: list[Message]


@dataclass
class MessagingPatterns:
    max_burst_len: float
    mean_burst_len: float
    median_response_latency: float
    mean_response_latency: float


@dataclass
class EmojiUse:
    emoji_frequency: float
    most_common_emojis: list[tuple[str, int]]


@dataclass
class Lexicon:
    distinctive_words: list[tuple[str, float]]
    distinctive_phrases: list[tuple[str, float]]


@dataclass
class CommunicationStyle:
    avg_message_len: float
    emoji_use: EmojiUse
    messaging_patterns: MessagingPatterns
    lexicon: Lexicon
    sample_messages: list[Message]


@dataclass
class BehavioralPatterns:
    # TODO
    # idk what this is
    claims: list[str] = field(default_factory=list)


type Username = str
