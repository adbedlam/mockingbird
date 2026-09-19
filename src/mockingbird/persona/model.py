from dataclasses import dataclass, field


@dataclass
class CommunicationStyle:
    avg_message_len: float
    emoji_frequency: float
    most_common_emojis: list[tuple[str, int]]
    burst_stats: tuple[float, float]  # max, mean
    response_latency_stats: tuple[float, float]  # median, mean в секах
    distinctive_words: list[tuple[str, float]]
    distinctive_phrases: list[tuple[str, float]]
    sample_messages: list[str]


@dataclass
class BehavioralPatterns:
    # TODO
    claims: list[str] = field(default_factory=list)


@dataclass
class Persona:
    user_id: int
    message_count: int
    communication_style: CommunicationStyle
    behavior_patterns: BehavioralPatterns = field(default_factory=BehavioralPatterns)

    def describe_length(sel, avg_len: float) -> str:
        if avg_len < 20:
            return "обычно пишет очень коротко"
        if avg_len < 60:
            return "пишет сообщения средней длины"
        return "часто пишет развёрнуто"

    def to_system_prompt(self) -> str:
        style = self.communication_style
        words = ", ".join(w for w, _ in style.distinctive_words[:5])
        phrases = ", ".join([p for p, _ in style.distinctive_phrases[:5]])
        emojis = "".join(e for e, _ in style.most_common_emojis)

        return "sys prompt"
