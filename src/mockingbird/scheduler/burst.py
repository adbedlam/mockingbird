import random
from collections import Counter
from datetime import datetime, timedelta

import numpy as np

from mockingbird.conversation.history import History
from mockingbird.scheduler.distribution import Distribution


class Burst:
    def __init__(
        self,
        history: History,
        burst_window: timedelta = timedelta(minutes=10),
        delay_bin_size: timedelta = timedelta(seconds=1),
        smoothing_sigma: timedelta = timedelta(seconds=2),
    ) -> None:
        self.burst_window = burst_window
        self.delay_bin_size = delay_bin_size
        self.smoothing_sigma = smoothing_sigma

        self.bursts = self._get_bursts(history)
        self.size_distribution = self._build_size_distribution()
        self.delay_distribution = self._build_delay_distribution()

    def _get_bursts(
        self,
        history: History,
    ) -> list[list[datetime]]:
        messages = sorted(history.messages, key=lambda message: message.timestamp)

        bursts: list[list[datetime]] = []
        current: list[datetime] = []

        for message in messages:
            if message.author != history.user:
                if len(current) > 1:
                    bursts.append(current)
                current = []
                continue

            timestamp = message.timestamp

            if current and timestamp - current[-1] > self.burst_window:
                if current:
                    bursts.append(current)
                current = []

            current.append(timestamp)

        if current:
            bursts.append(current)

        return bursts

    def _build_size_distribution(self) -> Distribution:
        counts = Counter(len(burst) for burst in self.bursts)

        if not counts:
            return Distribution([], [])

        values = sorted(counts)
        total = sum(counts.values())

        return Distribution(
            [counts[value] / total for value in values],
            values,
        )

    def _build_delay_distribution(self) -> Distribution:
        delays = [
            (current - previous).total_seconds()
            for burst in self.bursts
            for previous, current in zip(burst, burst[1:])
        ]

        if not delays:
            return Distribution([], [])

        bin_width = self.delay_bin_size.total_seconds()
        sigma = self.smoothing_sigma.total_seconds() / bin_width

        indices = [int(delay / bin_width) for delay in delays]
        histogram = np.bincount(indices).astype(float)

        radius = max(1, int(np.ceil(3 * sigma)))
        x = np.arange(-radius, radius + 1)

        kernel = np.exp(-0.5 * (x / sigma) ** 2)
        kernel /= kernel.sum()

        padded = np.pad(histogram, radius, mode="edge")
        smoothed = np.convolve(padded, kernel, mode="valid")

        probabilities = smoothed / smoothed.sum()

        values = [
            timedelta(seconds=(i + 0.5) * bin_width) for i in range(len(probabilities))
        ]

        return Distribution(
            probabilities.tolist(),
            values,
        )

    def generate(self) -> list[timedelta]:
        if not self.bursts:
            return []

        size = random.choices(
            self.size_distribution.values,
            weights=self.size_distribution.probabilities,
            k=1,
        )[0]

        delays = random.choices(
            self.delay_distribution.values,
            weights=self.delay_distribution.probabilities,
            k=size - 1,
        )

        return delays
