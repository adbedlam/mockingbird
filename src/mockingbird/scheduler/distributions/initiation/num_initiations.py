from collections import Counter, defaultdict
from datetime import timedelta

from mockingbird.conversation.history import History
from mockingbird.conversation.message_type import MessageType
from mockingbird.scheduler.distributions.distribution import Distribution
from mockingbird.scheduler.distributions.initiation.get_initiations import get_initiations
from mockingbird.scheduler.weekday import Weekday


class NumInitiations:
    def __init__(self, history: History):
        daily_counts = defaultdict(list)

        messages = history.messages

        if not messages:
            raise ValueError("Cannot build NumInitiations from empty history")

        start_date = messages[0].timestamp.date()
        end_date = messages[-1].timestamp.date()

        counts = Counter()

        previous = None

        for message in get_initiations(history):
            counts[message.timestamp.date()] += 1

        current_date = start_date

        while current_date <= end_date:
            weekday = Weekday(current_date.weekday() + 1)
            daily_counts[weekday].append(counts[current_date])
            current_date += timedelta(days=1)

        self.distributions = {}

        for weekday, values in daily_counts.items():
            frequencies = Counter(values)
            outcomes = sorted(frequencies)
            probabilities = [
                frequencies[value] / len(values)
                for value in outcomes
            ]

            self.distributions[weekday] = Distribution(
                values=outcomes,
                probabilities=probabilities,
            )

    def sample(self, weekday: Weekday) -> int:
        return self.distributions[weekday].sample()

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

    num_initiations = NumInitiations(history)

    fig, axes = plt.subplots(4, 2, figsize=(12, 14))
    axes = axes.flatten()

    for weekday, ax in zip(Weekday, axes):
        distribution = num_initiations.distributions.get(weekday)
        ax.set_title(weekday.name)

        if distribution is None:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            continue

        ax.bar(distribution.values, distribution.probabilities, width=0.8)
        ax.set_xlabel("Number of initiations")
        ax.set_ylabel("Probability")
        ax.set_xticks(distribution.values)
        ax.grid(axis="y", alpha=0.3)

    axes[-1].axis("off")
    fig.suptitle("Daily Initiations by Weekday")
    fig.tight_layout()
    plt.show()