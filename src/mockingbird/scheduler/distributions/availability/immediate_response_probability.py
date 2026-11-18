from collections import defaultdict
from datetime import datetime, timedelta

from mockingbird.conversation.history import History
from mockingbird.scheduler.observations import response_observations
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

        immediate: defaultdict[tuple[Weekday, int], int] = defaultdict(int)
        total: defaultdict[tuple[Weekday, int], int] = defaultdict(int)
        weekday_immediate: defaultdict[Weekday, int] = defaultdict(int)
        weekday_total: defaultdict[Weekday, int] = defaultdict(int)

        for incoming, delay in response_observations(history):
            timestamp = incoming.timestamp
            weekday = Weekday(timestamp.weekday() + 1)
            seconds = timestamp.hour * 3600 + timestamp.minute * 60 + timestamp.second
            key = (weekday, int(seconds // bin_seconds))
            total[key] += 1
            weekday_total[weekday] += 1
            if delay <= timedelta(minutes=10):
                immediate[key] += 1
                weekday_immediate[weekday] += 1

        total_responses = sum(weekday_total.values())

        if total_responses == 0:
            raise ValueError("No classified responses found in history")

        self.global_probability = sum(weekday_immediate.values()) / total_responses

        self.probabilities: dict[Weekday, list[float]] = {}

        for weekday in Weekday:
            probabilities = []

            for bucket in range(self.num_bins):
                key = (weekday, bucket)

                if total[key]:
                    probability = immediate[key] / total[key]
                elif weekday_total[weekday]:
                    probability = weekday_immediate[weekday] / weekday_total[weekday]
                else:
                    probability = self.global_probability

                probabilities.append(probability)

            self.probabilities[weekday] = probabilities

    def probability(self, timestamp: datetime) -> float:
        weekday = Weekday(timestamp.weekday() + 1)

        seconds = timestamp.hour * 3600 + timestamp.minute * 60 + timestamp.second
        bucket = int(seconds // self.bin_seconds)

        return self.probabilities[weekday][bucket]
