import random
from collections import Counter
from datetime import timedelta

from mockingbird.conversation.history import History
from mockingbird.scheduler.distributions.distribution import Distribution, smooth_distribution
from mockingbird.scheduler.distributions.in_conversation.utils import get_conversation_bursts


class ConversationBurstDelay:
    def __init__(
        self,
        history: History,
        sigma: float = 100,
    ):
        delays = [
            (message.timestamp - previous.timestamp).total_seconds()
            for burst in get_conversation_bursts(history)
            for previous, message in zip(burst, burst[1:])
        ]

        if not delays:
            raise ValueError("No conversation burst delays found in history")

        counts = Counter(delays)
        values = sorted(counts)

        self.distribution = smooth_distribution(
            Distribution(
                values=values,
                probabilities=[counts[value] / len(delays) for value in values],
            ),
            sigma=sigma,
            circular=False,
        )

    def sample(self) -> timedelta:
        delay = self.distribution.sample()
        return timedelta(seconds=delay + random.uniform(0, 1))

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

    burst_delay = ConversationBurstDelay(history)

    values = burst_delay.distribution.values
    probabilities = burst_delay.distribution.probabilities

    mean = sum(
        value * probability
        for value, probability in zip(values, probabilities)
    )

    variance = sum(
        probability * (value - mean) ** 2
        for value, probability in zip(values, probabilities)
    )

    std = variance ** 0.5

    print("Conversation Burst Delay Statistics")
    print(f"Mean:     {mean:.2f} seconds")
    print(f"Std dev:  {std:.2f} seconds")
    print(f"Minimum:  {min(values):.2f} seconds")
    print(f"Maximum:  {max(values):.2f} seconds")
    print(f"Median:   {values[sorted(range(len(values)), key=lambda i: values[i])[next(
        i for i, p in enumerate(
            __import__('itertools').accumulate(probabilities)
        ) if p >= 0.5
    )]]:.2f} seconds")

    plt.bar(values, probabilities, width=0.8)
    # plt.axvline(mean, linestyle="--", label=f"Mean: {mean:.2f}s")
    plt.xlabel("Delay between messages (seconds)")
    plt.ylabel("Probability")
    plt.title("Conversation Burst Delay Distribution")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()