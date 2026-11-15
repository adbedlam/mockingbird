from mockingbird.conversation.history import History
from mockingbird.scheduler.distributions.distribution import Distribution
from mockingbird.scheduler.distributions.initiation.get_initiations import get_initiation_bursts


class InitiationBurstLength:
    def __init__(self, history: History):
        values = [
            len(burst)
            for burst in get_initiation_bursts(history)
        ]

        if not values:
            raise ValueError("No initiation bursts found in history")

        self.distribution = Distribution(
            values=values,
            probabilities=[1 / len(values)] * len(values),
        )

    def sample(self) -> int:
        return self.distribution.sample()

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from collections import Counter
    from pathlib import Path

    from mockingbird.config import load_config
    from mockingbird.conversation.history import History

    config = load_config()
    history_path = (
        Path(config["paths"]["processed"])
        / config["paths"]["history_filename"]
    )
    history = History.load(history_path)

    burst_length = InitiationBurstLength(history)
    frequencies = Counter(burst_length.distribution.values)

    values = sorted(frequencies)
    counts = [frequencies[value] for value in values]

    plt.bar(values, counts)
    plt.xticks(values)
    plt.xlabel("Messages per initiation burst")
    plt.ylabel("Frequency")
    plt.title("Initiation Burst Length")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()