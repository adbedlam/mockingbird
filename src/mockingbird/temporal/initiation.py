import random
from collections import Counter, defaultdict
from datetime import datetime, time, timedelta

import numpy as np

from mockingbird.conversation.history import History
from mockingbird.temporal.distribution import Distribution
from mockingbird.temporal.weekday import Weekday


class Initiation:
    def __init__(
        self,
        history: History,
        bin_size: timedelta,
        conversation_window: timedelta,
    ) -> None:
        self.bin_size = bin_size
        self.conversation_window = conversation_window

        initial_times = self._get_initial_times(history)

        self.count_distribution = self._build_count_distribution(history, initial_times)
        self.time_distribution = self._build_time_distribution(initial_times)

    def _get_initial_times(
        self,
        history: History,
    ) -> dict[Weekday, list[datetime]]:
        initial_times = {day: [] for day in Weekday}
        previous_timestamp = None

        for message in history.messages:
            timestamp = message.timestamp

            ongoing = (
                previous_timestamp is not None
                and timestamp - previous_timestamp <= self.conversation_window
            )

            if message.author == history.user and not ongoing:
                initial_times[Weekday(timestamp.weekday())].append(timestamp)

            previous_timestamp = timestamp

        return initial_times

    def _build_count_distribution(
        self,
        history: History,
        initial_times: dict[Weekday, list[datetime]],
    ) -> dict[Weekday, Distribution]:
        if not history.messages:
            return {day: Distribution([], []) for day in Weekday}

        daily_counts = defaultdict(Counter)

        for day, timestamps in initial_times.items():
            for timestamp in timestamps:
                daily_counts[day][timestamp.date()] += 1

        dates = [message.timestamp.date() for message in history.messages]

        frequencies = {day: Counter() for day in Weekday}

        current_date = min(dates)
        last_date = max(dates)

        while current_date <= last_date:
            weekday = Weekday(current_date.weekday())
            count = daily_counts[weekday][current_date]

            frequencies[weekday][count] += 1
            current_date += timedelta(days=1)

        distributions = {}

        for day, counts in frequencies.items():
            values = list(counts)
            total = sum(counts.values())

            distributions[day] = Distribution(
                probabilities=[counts[value] / total for value in values],
                values=values,
            )

        return distributions

    def _build_time_distribution(
        self,
        initial_times: dict[Weekday, list[datetime]],
    ) -> dict[Weekday, Distribution]:
        day_seconds = timedelta(days=1).total_seconds()
        bin_seconds = self.bin_size.total_seconds()

        if bin_seconds <= 0 or day_seconds % bin_seconds != 0:
            raise ValueError("bin_size must be positive and divide one day evenly")

        bins = int(day_seconds / bin_seconds)

        distributions = {day: np.zeros(bins) for day in Weekday}

        for day, timestamps in initial_times.items():
            for timestamp in timestamps:
                minute = timestamp.hour * 60 + timestamp.minute
                bin_index = int(timedelta(minutes=minute) / self.bin_size)
                distributions[day][bin_index] += 1

        # Circular Gaussian smoothing.
        sigma = timedelta(minutes=15).total_seconds() / bin_seconds
        indices = np.arange(bins)
        distances = np.minimum(indices, bins - indices)

        kernel = np.exp(-0.5 * (distances / sigma) ** 2)
        kernel /= kernel.sum()

        result = {}

        for day, counts in distributions.items():
            smoothed = np.fft.ifft(np.fft.fft(counts) * np.fft.fft(kernel)).real

            smoothed = np.maximum(smoothed, 0)
            total = smoothed.sum()

            probabilities = (smoothed / total).tolist() if total > 0 else [0.0] * bins

            result[day] = Distribution(
                probabilities=probabilities,
                values=list(range(bins)),
            )

        return result

    def _generate_count(self, weekday: Weekday) -> int:
        distribution = self.count_distribution[weekday]

        if not distribution.values:
            return 0

        return random.choices(
            distribution.values,
            weights=distribution.probabilities,
            k=1,
        )[0]

    def _generate_times(
        self,
        weekday: Weekday,
        count: int,
    ) -> list[time]:
        if count <= 0:
            return []

        distribution = self.time_distribution[weekday]

        available = {
            index: probability
            for index, probability in zip(
                distribution.values,
                distribution.probabilities,
            )
            if probability > 0
        }

        selected_times = []
        bin_seconds = self.bin_size.total_seconds()
        bins = len(distribution.values)

        while count > 0 and available:
            indices = list(available)
            weights = list(available.values())

            selected = random.choices(
                indices,
                weights=weights,
                k=1,
            )[0]

            selected_time = (
                datetime.min
                + self.bin_size * selected
                + timedelta(seconds=random.random() * bin_seconds)
            ).time()

            selected_times.append(selected_time)
            count -= 1

            available = {
                index: probability
                for index, probability in available.items()
                if min(
                    (index - selected) % bins,
                    (selected - index) % bins,
                )
                * bin_seconds
                >= self.conversation_window.total_seconds()
            }

        return sorted(selected_times)

    def generate(self, weekday: Weekday) -> list[time]:
        count = self._generate_count(weekday)
        return self._generate_times(weekday, count)
