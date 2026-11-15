from collections import Counter
from datetime import timedelta
import random

from mockingbird.conversation.history import History
from mockingbird.scheduler.distributions.distribution import (
    Distribution,
    smooth_distribution,
)
from mockingbird.scheduler.distributions.initiation.get_initiations import (
    get_initiation_bursts,
)


class InitiationBurstDelay:
    def __init__(
        self,
        history: History,
        sigma: float = 4,
    ):
        delays = [
            (message.timestamp - previous.timestamp).total_seconds()
            for burst in get_initiation_bursts(history)
            for previous, message in zip(burst, burst[1:])
        ]

        if not delays:
            raise ValueError("No initiation burst delays found in history")

        counts = Counter(delays)
        values = sorted(counts)

        distribution = Distribution(
            values=values,
            probabilities=[counts[value] / len(delays) for value in values],
        )

        self.distribution = smooth_distribution(
            distribution,
            sigma=sigma,
            circular=False,
        )

        self.distribution.values = [
            timedelta(seconds=value)
            for value in self.distribution.values
        ]

    def sample(self) -> timedelta:
        delay = self.distribution.sample()
        return delay + timedelta(seconds=random.uniform(0, 1))


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

    burst_delay = InitiationBurstDelay(history)

    delays = [
        delay.total_seconds()
        for delay in burst_delay.distribution.values
    ]

    plt.bar(
        delays,
        burst_delay.distribution.probabilities,
        width=0.8,
    )
    plt.xlabel("Delay between messages (seconds)")
    plt.ylabel("Probability")
    plt.title("Initiation Burst Delay Distribution")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()