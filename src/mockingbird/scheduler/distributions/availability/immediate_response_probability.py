from collections import defaultdict
from datetime import datetime, timedelta

from mockingbird.conversation.history import History
from mockingbird.conversation.message_type import MessageType
from mockingbird.scheduler.weekday import Weekday


class ImmediateResponseProbability:
    def __init__(
        self,
        history: History,
        bin_size: timedelta = timedelta(minutes=30),
    ):
        bin_seconds = bin_size.total_seconds()

        if bin_seconds <= 0 or bin_seconds % 60 != 0:
            raise ValueError("bin_size must be a positive whole number of minutes")

        if 86400 % bin_seconds != 0:
            raise ValueError("bin_size must divide 24 hours evenly")

        self.num_bins = int(86400 // bin_seconds)
        self.bin_seconds = bin_seconds

        immediate = defaultdict(int)
        total = defaultdict(int)
        weekday_immediate = defaultdict(int)
        weekday_total = defaultdict(int)

        messages = history.messages
        index = 0

        while index < len(messages):
            message = messages[index]

            is_incoming_initiation = (
                message.author_id != history.user_id
                and message.message_type == MessageType.INITIATION
            )

            if not is_incoming_initiation:
                index += 1
                continue

            burst = [message]
            index += 1

            while index < len(messages):
                current = messages[index]

                if (
                    current.author_id == message.author_id
                    and current.message_type == MessageType.INITIATION
                ):
                    burst.append(current)
                    index += 1
                else:
                    break

            response = None

            while index < len(messages):
                current = messages[index]

                if current.author_id == history.user_id:
                    response = current
                    break

                index += 1

            if response is None:
                break

            if response.message_type in (
                MessageType.IMMEDIATE_RESPONSE,
                MessageType.DELAYED_RESPONSE,
            ):
                timestamp = burst[-1].timestamp
                weekday = Weekday(timestamp.weekday() + 1)

                seconds = (
                    timestamp.hour * 3600
                    + timestamp.minute * 60
                    + timestamp.second
                )
                bucket = int(seconds // bin_seconds)

                key = (weekday, bucket)
                total[key] += 1
                weekday_total[weekday] += 1

                if response.message_type == MessageType.IMMEDIATE_RESPONSE:
                    immediate[key] += 1
                    weekday_immediate[weekday] += 1

            index += 1

        total_responses = sum(weekday_total.values())

        if total_responses == 0:
            raise ValueError("No classified responses found in history")

        self.global_probability = (
            sum(weekday_immediate.values()) / total_responses
        )

        self.probabilities: dict[Weekday, list[float]] = {}

        for weekday in Weekday:
            probabilities = []

            for bucket in range(self.num_bins):
                key = (weekday, bucket)

                if total[key]:
                    probability = immediate[key] / total[key]
                elif weekday_total[weekday]:
                    probability = (
                        weekday_immediate[weekday] / weekday_total[weekday]
                    )
                else:
                    probability = self.global_probability

                probabilities.append(probability)

            self.probabilities[weekday] = probabilities

    def probability(self, timestamp: datetime) -> float:
        weekday = Weekday(timestamp.weekday() + 1)

        seconds = (
            timestamp.hour * 3600
            + timestamp.minute * 60
            + timestamp.second
        )
        bucket = int(seconds // self.bin_seconds)

        return self.probabilities[weekday][bucket]

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

    distribution = ImmediateResponseProbability(history)

    fig, axes = plt.subplots(4, 2, figsize=(14, 12))
    axes = axes.flatten()

    labels = [
        f"{int(i * distribution.bin_seconds // 3600):02d}:"
        f"{int(i * distribution.bin_seconds % 3600 // 60):02d}"
        for i in range(distribution.num_bins)
    ]

    for weekday, ax in zip(Weekday, axes):
        probabilities = distribution.probabilities[weekday]

        ax.plot(labels, probabilities)
        ax.set_title(weekday.name)
        ax.set_xlabel("Time of day")
        ax.set_ylabel("Immediate response probability")
        ax.set_ylim(0, 1)
        ax.set_xticks(range(0, len(labels), max(1, len(labels) // 8)))
        ax.set_xticklabels(
            labels[::max(1, len(labels) // 8)],
            rotation=45,
        )
        ax.grid(alpha=0.3)

    axes[-1].axis("off")
    fig.suptitle("Immediate Response Probability by Weekday")
    fig.tight_layout()
    plt.show()