import random
from datetime import timedelta

import numpy as np

from mockingbird.conversation.history import History
from mockingbird.temporal.distribution import Distribution


class ResponseDelay:
    def __init__(
        self,
        history: History,
        response_window: timedelta = timedelta(minutes=10),
        bin_size: timedelta = timedelta(seconds=1),
        smoothing_sigma: timedelta = timedelta(seconds=2),
    ) -> None:
        self.response_window = response_window
        self.bin_size = bin_size
        self.smoothing_sigma = smoothing_sigma

        self.delays = self._get_delays(history)
        self.distribution = self._build_distribution()

    def _get_delays(self, history: History) -> list[timedelta]:
        messages = sorted(history.messages, key=lambda m: m.timestamp)

        delays = []
        last_other_message = None

        for message in messages:
            if message.author != history.user:
                last_other_message = message.timestamp
                continue

            if last_other_message is None:
                continue

            delay = message.timestamp - last_other_message

            if timedelta(0) < delay <= self.response_window:
                delays.append(delay)

            last_other_message = None

        return delays

    def _build_distribution(self) -> Distribution:
        if not self.delays:
            return Distribution([], [])

        bin_width = self.bin_size.total_seconds()
        sigma = self.smoothing_sigma.total_seconds() / bin_width

        indices = [int(delay.total_seconds() / bin_width) for delay in self.delays]

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
            probabilities=probabilities.tolist(),
            values=values,
        )

    def generate(self) -> timedelta:
        if not self.distribution.values:
            raise ValueError("Cannot generate response delay: no samples available")

        return random.choices(
            self.distribution.values,
            weights=self.distribution.probabilities,
            k=1,
        )[0]
