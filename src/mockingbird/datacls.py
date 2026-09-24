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

@dataclass
class Persona:
    communication_styles: dict[Username, CommunicationStyle]
    behavioral_patterns: BehavioralPatterns = field(default_factory=BehavioralPatterns)

    def describe_length(self, avg_len: float) -> str:
        if avg_len < 20:
            return "обычно пишет очень коротко"
        if avg_len < 60:
            return "пишет сообщения средней длины"
        return "часто пишет развёрнуто"

    def to_system_prompt(self, username: str) -> str:
        style = self.communication_styles[username]
        words = ", ".join(w for w, _ in style.distinctive_words[:5])
        phrases = ", ".join([p for p, _ in style.distinctive_phrases[:5]])
        emojis = "".join(e for e, _ in style.most_common_emojis)

        return "sys prompt"
