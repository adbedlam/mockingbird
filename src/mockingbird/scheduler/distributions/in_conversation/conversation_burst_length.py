from collections import Counter

from mockingbird.conversation.history import History
from mockingbird.scheduler.distributions.distribution import Distribution
from mockingbird.scheduler.distributions.in_conversation.utils import get_conversation_bursts


class ConversationBurstLength:
    def __init__(self, history: History):
        values = [
            len(burst)
            for burst in get_conversation_bursts(history)
        ]

        if not values:
            raise ValueError("No conversation bursts found in history")

        counts = Counter(values)
        outcomes = sorted(counts)

        self.distribution = Distribution(
            values=outcomes,
            probabilities=[counts[value] / len(values) for value in outcomes],
        )

    def sample(self) -> int:
        return self.distribution.sample()


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from pathlib import Path

    from mockingbird.config import load_config

    config = load_config()
    history_path = (
        Path(config["paths"]["processed"])
        / config["paths"]["history_filename"]
    )
    history = History.load(history_path)

    burst_length = ConversationBurstLength(history)

    plt.bar(
        burst_length.distribution.values,
        burst_length.distribution.probabilities,
        width=0.8,
    )
    plt.xlabel("Messages per conversation burst")
    plt.ylabel("Probability")
    plt.title("Conversation Burst Length")
    plt.xticks(burst_length.distribution.values)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()