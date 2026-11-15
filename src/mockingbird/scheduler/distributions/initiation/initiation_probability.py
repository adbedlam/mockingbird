
from datetime import time, timedelta

from mockingbird.conversation.history import History
from mockingbird.scheduler.distributions.distribution import (
    Distribution,
    smooth_distribution,
)
from mockingbird.scheduler.weekday import Weekday
from mockingbird.scheduler.distributions.initiation.get_initiations import get_initiations

class InitiationProbability:
    def __init__(
        self,
        history: History,
        bin_size: timedelta = timedelta(minutes=30),
        sigma: float = 1.5,
    ):
        bin_seconds = bin_size.total_seconds()

        if bin_seconds <= 0 or bin_seconds % 60 != 0:
            raise ValueError("bin_size must be a positive whole number of minutes")

        if 86400 % bin_seconds != 0:
            raise ValueError("bin_size must divide 24 hours evenly")

        num_bins = int(86400 // bin_seconds)
        num_week_bins = 7 * num_bins

        weekly_counts = [0] * num_week_bins

        for message in get_initiations(history):
            weekday = message.timestamp.weekday()

            seconds = (
                message.timestamp.hour * 3600
                + message.timestamp.minute * 60
                + message.timestamp.second
            )

            bucket = int(seconds // bin_seconds)
            weekly_index = weekday * num_bins + bucket

            weekly_counts[weekly_index] += 1

        weekly_distribution = smooth_distribution(
            Distribution(
                values=list(range(num_week_bins)),
                probabilities=weekly_counts,
            ),
            sigma=sigma,
            circular=True,
        )

        self.distributions: dict[Weekday, Distribution[time]] = {}

        for weekday in Weekday:
            start = (weekday.value - 1) * num_bins
            end = start + num_bins

            probabilities = weekly_distribution.probabilities[start:end]
            total = sum(probabilities)

            if total == 0:
                continue

            values = [
                time(
                    hour=int(bucket * bin_seconds // 3600),
                    minute=int(bucket * bin_seconds % 3600 // 60),
                )
                for bucket in range(num_bins)
            ]

            self.distributions[weekday] = Distribution(
                values=values,
                probabilities=[p / total for p in probabilities],
            )

    def sample(self, weekday: Weekday) -> time:
        if weekday not in self.distributions:
            raise ValueError(
                f"No initiation observations for {weekday.name}"
            )

        return self.distributions[weekday].sample()

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from pathlib import Path

    from mockingbird.config import load_config
    from mockingbird.conversation.history import History

    config = load_config()

    history_path = (
        Path(config["paths"]["processed"])
        / config["paths"]["history_filename"]
    )

    history = History.load(history_path)
    initiation_probability = InitiationProbability(history)

    fig, axes = plt.subplots(4, 2, figsize=(14, 12))
    axes = axes.flatten()

    for weekday, ax in zip(Weekday, axes):
        distribution = initiation_probability.distributions.get(weekday)

        ax.set_title(weekday.name)

        if distribution is None:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            continue

        labels = [value.strftime("%H:%M") for value in distribution.values]

        ax.bar(labels, distribution.probabilities, width=0.9)
        ax.set_xlabel("Time of day")
        ax.set_ylabel("Probability")
        ax.set_xticks(range(0, len(labels), 4))
        ax.set_xticklabels(labels[::4])
        ax.grid(axis="y", alpha=0.3)

    axes[-1].axis("off")

    fig.suptitle("Initiation Probability by Weekday")
    fig.tight_layout()
    plt.show()