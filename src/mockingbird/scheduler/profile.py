from __future__ import annotations

from collections import Counter
from dataclasses import replace
from datetime import date, datetime, time, timedelta
from random import Random
from zoneinfo import ZoneInfo

from mockingbird.conversation.history import History
from mockingbird.scheduler.observations import (
    conversation_bursts,
    response_observations,
)

SCHEDULER_SETTINGS = {
    "timezone",
    "reply_mode",
    "max_initiations_per_day",
    "bin_minutes",
    "activity_coverage",
    "min_weekday_messages",
    "min_initiation_gap_seconds",
    "silence_before_initiation_seconds",
    "conversation_window",
    "min_reply_delay_seconds",
    "max_reply_delay_seconds",
    "max_lateness_seconds",
}


def validate_scheduler_settings(settings: dict) -> None:
    if not isinstance(settings, dict):
        raise ValueError("scheduler settings must be a mapping")
    unknown = set(settings) - SCHEDULER_SETTINGS
    if unknown:
        raise ValueError("Unknown scheduler settings: " + ", ".join(sorted(unknown)))
    missing = SCHEDULER_SETTINGS - set(settings)
    if missing:
        raise ValueError("Missing scheduler settings: " + ", ".join(sorted(missing)))
    ZoneInfo(settings["timezone"])
    if settings["reply_mode"] not in ("history", "immediate"):
        raise ValueError("reply_mode must be history or immediate")
    if settings["max_initiations_per_day"] < 0:
        raise ValueError("max_initiations_per_day cannot be negative")
    if settings["bin_minutes"] <= 0 or 1440 % settings["bin_minutes"]:
        raise ValueError("bin_minutes must divide 1440 evenly")
    if not 0 < settings["activity_coverage"] <= 1:
        raise ValueError("activity_coverage must be in (0, 1]")
    for key in (
        "min_weekday_messages",
        "min_initiation_gap_seconds",
        "silence_before_initiation_seconds",
        "conversation_window",
        "min_reply_delay_seconds",
        "max_reply_delay_seconds",
        "max_lateness_seconds",
    ):
        if settings[key] <= 0:
            raise ValueError(f"{key} must be positive")
    if settings["max_reply_delay_seconds"] < settings["min_reply_delay_seconds"]:
        raise ValueError("max_reply_delay_seconds must be >= minimum delay")


class BehaviorProfile:
    """Fit only on real pair history, never on generated chat messages."""

    def __init__(self, history: History, config: dict):
        validate_scheduler_settings(config)
        config = dict(config)
        self.config = config
        self.zone = ZoneInfo(config["timezone"])
        messages = sorted(
            (
                replace(m, timestamp=self.local(m.timestamp, allow_naive=True))
                for m in history.messages
                if m.author in (history.user, history.other)
            ),
            key=lambda m: m.timestamp,
        )
        modeled = [m for m in messages if m.author == history.user]
        if not modeled:
            raise ValueError("Scheduler needs real messages from the modeled person")
        normalized = History(history.user, history.other, messages)
        bins = 1440 // config["bin_minutes"]
        self.activity: dict[int, list[float]] = {}
        self.initiation_weights: dict[int, list[float]] = {}
        self.daily_counts: dict[int, list[int]] = {day: [] for day in range(7)}
        global_counts = [0.0] * bins
        weekday_counts = [[0.0] * bins for _ in range(7)]
        for message in modeled:
            bucket = self.bucket(message.timestamp)
            global_counts[bucket] += 1
            weekday_counts[message.timestamp.weekday()][bucket] += 1
        for day in range(7):
            counts = weekday_counts[day]
            if sum(counts) < config["min_weekday_messages"]:
                counts = global_counts
            # Keep bins accounting for 90% of observed messages. No assumption
            # of conventional bedtime: a night owl can be active after midnight.
            selected: set[int] = set()
            accumulated = 0.0
            for bucket in sorted(range(bins), key=lambda i: counts[i], reverse=True):
                if accumulated >= sum(counts) * config["activity_coverage"]:
                    break
                selected.add(bucket)
                accumulated += counts[bucket]
            self.activity[day] = [
                counts[i] if i in selected else 0.0 for i in range(bins)
            ]

        bursts = list(
            conversation_bursts(
                normalized, timedelta(seconds=config["conversation_window"])
            )
        )
        initiations = [
            b.messages[0] for b in bursts if b.author == history.user and b.initiated
        ]
        self.observed_initiations = len(initiations)
        self.message_count = len(modeled)
        self.observed_days = (
            messages[-1].timestamp.date() - messages[0].timestamp.date()
        ).days + 1
        counts_per_date = Counter(m.timestamp.date() for m in initiations)
        for offset in range(self.observed_days):
            current = messages[0].timestamp.date() + timedelta(days=offset)
            self.daily_counts[current.weekday()].append(counts_per_date[current])
        global_initiations = [0.0] * bins
        weekday_initiations = [[0.0] * bins for _ in range(7)]
        for message in initiations:
            bucket = self.bucket(message.timestamp)
            global_initiations[bucket] += 1
            weekday_initiations[message.timestamp.weekday()][bucket] += 1
        for day in range(7):
            weights = weekday_initiations[day]
            if sum(weights) < 5:
                weights = global_initiations
            # Schedule only within observed active bins, even if an initiative
            # observation itself was a rare overnight outlier.
            self.initiation_weights[day] = [
                weights[i] if self.activity[day][i] else 0.0 for i in range(bins)
            ]
        self.reply_delays = [
            delay.total_seconds()
            for _, delay in response_observations(
                normalized,
                timedelta(seconds=config["conversation_window"]),
                max_delay=timedelta(hours=24),
            )
        ]

    def local(self, timestamp: datetime, *, allow_naive: bool = False) -> datetime:
        if timestamp.tzinfo is None or timestamp.utcoffset() is None:
            if not allow_naive:
                raise ValueError("Runtime timestamps must include a timezone")
            timestamp = timestamp.replace(tzinfo=self.zone)
        return timestamp.astimezone(self.zone)

    def bucket(self, timestamp: datetime) -> int:
        return (timestamp.hour * 60 + timestamp.minute) // self.config["bin_minutes"]

    def is_active(self, timestamp: datetime) -> bool:
        local = self.local(timestamp)
        return self.activity[local.weekday()][self.bucket(local)] > 0

    def next_active(self, timestamp: datetime) -> datetime:
        current = self.local(timestamp)
        for _ in range(7 * 1440 // self.config["bin_minutes"] + 1):
            if self.is_active(current):
                return current
            minute = self.bucket(current) * self.config["bin_minutes"]
            current = current.replace(
                hour=minute // 60, minute=minute % 60, second=0, microsecond=0
            )
            current += timedelta(minutes=self.config["bin_minutes"])
        raise ValueError("No active time found in the weekly profile")

    def plan(self, day: date, rng: Random) -> list[datetime]:
        day_counts = self.daily_counts[day.weekday()]
        # Sparse weekday samples fall back to the full observation period,
        # including days with zero initiatives.
        if len(day_counts) < 3:
            day_counts = [n for values in self.daily_counts.values() for n in values]
        count = min(rng.choice(day_counts), self.config["max_initiations_per_day"])
        weights = self.initiation_weights[day.weekday()]
        if not any(weights) or count == 0:
            return []
        planned: list[datetime] = []
        for _ in range(100):
            if len(planned) >= count:
                break
            bucket = rng.choices(range(len(weights)), weights=weights, k=1)[0]
            seconds = bucket * self.config["bin_minutes"] * 60 + rng.randrange(
                self.config["bin_minutes"] * 60
            )
            candidate = datetime.combine(day, time(), self.zone) + timedelta(
                seconds=seconds
            )
            if all(
                abs((candidate - other).total_seconds())
                >= self.config["min_initiation_gap_seconds"]
                for other in planned
            ):
                planned.append(candidate)
        return sorted(planned)

    def reply_delay(self, rng: Random) -> timedelta:
        seconds = rng.choice(self.reply_delays) if self.reply_delays else 60.0
        return timedelta(
            seconds=max(
                self.config["min_reply_delay_seconds"],
                min(seconds, self.config["max_reply_delay_seconds"]),
            )
        )
