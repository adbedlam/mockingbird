import random
from collections import Counter, defaultdict
from datetime import date, datetime, time, timedelta

from mockingbird.conversation.history import History
from mockingbird.scheduler.distribution import Distribution
from mockingbird.scheduler.weekday import Weekday


class Busy:
    def __init__(
        self,
        history: History,
        bin_size: timedelta = timedelta(minutes=15),
        conversation_window: timedelta = timedelta(minutes=10),
        max_busy_duration: timedelta = timedelta(hours=3),
        randomness_interval: timedelta = timedelta(minutes=30),
    ):
        self.bin_size = bin_size
        self.conversation_window = conversation_window
        self.max_busy_duration = max_busy_duration
        self.randomness_interval = randomness_interval

        self.unreplied = self._get_unreplied(history)

        self.count_distribution = self._build_count_distribution(history)
        self.time_distribution = self._build_time_distribution()
        self.length_distribution = self._build_length_distribution()

    def _get_unreplied(
        self,
        history: History,
    ) -> list[tuple[datetime, timedelta]]:
        messages = history.messages
        user_messages = [
            message.timestamp for message in messages if message.author == history.user
        ]

        incoming = [
            message.timestamp for message in messages if message.author != history.user
        ]

        if not incoming or not user_messages:
            return []

        episodes: list[list[datetime]] = []
        result: list[tuple[datetime, timedelta]] = []

        for timestamp in incoming:
            if not episodes or timestamp - episodes[-1][-1] > self.conversation_window:
                episodes.append([timestamp])
            else:
                episodes[-1].append(timestamp)

        for episode in episodes:
            start = episode[0]
            last = episode[-1]

            previous_replies = [
                timestamp for timestamp in user_messages if timestamp <= start
            ]
            next_replies = [
                timestamp for timestamp in user_messages if timestamp > last
            ]

            if previous_replies and (
                start - previous_replies[-1] <= self.conversation_window
            ):
                continue

            if not next_replies:
                continue

            reply = next_replies[0]

            if reply - last <= self.conversation_window:
                continue

            duration = min(
                reply - start,
                self.max_busy_duration,
            )

            result.append((start, duration))

        return result

    def _build_count_distribution(
        self,
        history: History,
    ) -> dict[Weekday, Distribution]:
        if not history.messages:
            return {}

        first_date = history.messages[0].timestamp.date()
        last_date = history.messages[-1].timestamp.date()

        counts: defaultdict[Weekday, Counter[date]] = defaultdict(Counter)

        for timestamp, _ in self.unreplied:
            counts[Weekday(timestamp.weekday())][timestamp.date()] += 1

        result = {}

        current_date = first_date
        while current_date <= last_date:
            weekday = Weekday(current_date.weekday())
            counts[weekday].setdefault(current_date, 0)
            current_date += timedelta(days=1)

        for weekday, daily_counts in counts.items():
            frequencies = Counter(daily_counts.values())
            values = sorted(frequencies)

            probabilities = [
                frequencies[value] / sum(frequencies.values()) for value in values
            ]

            result[weekday] = Distribution(probabilities, values)

        return result

    def _build_time_distribution(self) -> dict[Weekday, Distribution]:
        bins_per_day = int(timedelta(days=1) / self.bin_size)
        counts: defaultdict[Weekday, list[int]] = defaultdict(
            lambda: [0] * bins_per_day
        )

        for timestamp, _ in self.unreplied:
            weekday = Weekday(timestamp.weekday())
            bin_index = int(
                (timestamp.hour * 60 + timestamp.minute)
                / (self.bin_size.total_seconds() / 60)
            )
            counts[weekday][bin_index] += 1

        result = {}

        for weekday, frequencies in counts.items():
            total = sum(frequencies)

            probabilities = (
                [value / total for value in frequencies]
                if total
                else [1 / bins_per_day] * bins_per_day
            )

            result[weekday] = Distribution(
                probabilities,
                list(range(bins_per_day)),
            )

        return result

    def _build_length_distribution(self) -> dict[Weekday, Distribution]:
        counts: defaultdict[Weekday, Counter[int]] = defaultdict(Counter)

        for timestamp, duration in self.unreplied:
            weekday = Weekday(timestamp.weekday())

            length_bin = int(duration.total_seconds() / self.bin_size.total_seconds())
            counts[weekday][length_bin] += 1

        result = {}

        for weekday, frequencies in counts.items():
            values = sorted(frequencies)
            total = sum(frequencies.values())

            probabilities = [frequencies[value] / total for value in values]

            lengths = [(value + 0.5) * self.bin_size for value in values]

            result[weekday] = Distribution(probabilities, lengths)

        return result

    def _add_randomness(self, value: timedelta) -> timedelta:
        offset = random.uniform(
            -self.randomness_interval.total_seconds(),
            self.randomness_interval.total_seconds(),
        )
        return max(timedelta(0), value + timedelta(seconds=offset))

    def generate(
        self,
        weekday: Weekday,
    ) -> list[tuple[time, timedelta]]:
        count = random.choices(
            self.count_distribution[weekday].values,
            weights=self.count_distribution[weekday].probabilities,
            k=1,
        )[0]

        times = random.choices(
            self.time_distribution[weekday].values,
            weights=self.time_distribution[weekday].probabilities,
            k=count,
        )

        lengths = random.choices(
            self.length_distribution[weekday].values,
            weights=self.length_distribution[weekday].probabilities,
            k=count,
        )

        result = []

        for bin_index, duration in zip(times, lengths):
            offset = bin_index * self.bin_size
            jitter = random.uniform(0, self.bin_size.total_seconds())

            start = (
                datetime.combine(date.min, time.min)
                + offset
                + timedelta(seconds=jitter)
            ).time()

            duration = min(duration, self.max_busy_duration)
            duration = self._add_randomness(duration)

            result.append((start, duration))

        return sorted(result)
