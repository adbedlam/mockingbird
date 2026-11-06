import random
from datetime import datetime, time, timedelta

from mockingbird.temporal.distribution import Distribution
from mockingbird.temporal.weekday import Weekday


class Sleep:
    def __init__(
        self,
        initiation_distributions: dict[Weekday, Distribution],
        bin_size: timedelta,
        lower_percentile: float = 0.80,
        upper_percentile: float = 0.99,
        start_skew: tuple[float, float] = (2, 5),
        end_skew: tuple[float, float] = (2, 5),
        randomness_interval: timedelta = timedelta(minutes=30),
    ) -> None:
        self.start_skew = start_skew
        self.end_skew = end_skew
        self.randomness_interval = randomness_interval

        self.small_intervals = self._build_intervals(
            initiation_distributions, bin_size, lower_percentile
        )
        self.large_intervals = self._build_intervals(
            initiation_distributions, bin_size, upper_percentile
        )

    @staticmethod
    def _build_intervals(
        distributions: dict[Weekday, Distribution],
        bin_size: timedelta,
        coverage: float,
    ) -> dict[Weekday, tuple[time, time]]:
        result = {}

        for day, distribution in distributions.items():
            probabilities = distribution.probabilities
            bins = len(probabilities)
            extended = probabilities * 2

            best_start, best_width = 0, bins + 1
            end, probability = 0, 0.0

            for start in range(bins):
                while end < start + bins and probability < coverage:
                    probability += extended[end]
                    end += 1

                if probability >= coverage and end - start < best_width:
                    best_start = start
                    best_width = end - start

                probability -= extended[start]

            start_time = bin_size * best_start
            end_time = bin_size * (best_start + best_width)

            result[day] = (
                (datetime.min + start_time).time(),
                (datetime.min + end_time).time(),
            )

        return result

    @staticmethod
    def _sample_offset(
        gap: timedelta,
        skew: tuple[float, float],
    ) -> timedelta:
        minutes = gap.total_seconds() / 60

        if minutes <= 0:
            return timedelta(0)

        return timedelta(minutes=minutes * random.betavariate(*skew))

    def _add_randomness(self, value: time) -> time:
        offset = random.uniform(
            -self.randomness_interval.total_seconds(),
            self.randomness_interval.total_seconds(),
        )

        result = datetime.combine(datetime.min, value) + timedelta(seconds=offset)

        return result.time()

    def generate(self, weekday: Weekday) -> tuple[time, time]:
        large_start, large_end = self.large_intervals[weekday]
        small_start, small_end = self.small_intervals[weekday]

        large_start = datetime.combine(datetime.min, large_start)
        small_start = datetime.combine(datetime.min, small_start)
        small_end = datetime.combine(datetime.min, small_end)
        large_end = datetime.combine(datetime.min, large_end)

        start_gap = small_start - large_start
        end_gap = large_end - small_end

        start = (large_start + self._sample_offset(start_gap, self.start_skew)).time()

        end = (small_end + self._sample_offset(end_gap, self.end_skew)).time()

        start = self._add_randomness(start)
        end = self._add_randomness(end)

        return start, end
