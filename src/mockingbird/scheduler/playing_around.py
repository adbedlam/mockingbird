from datetime import datetime, timedelta
from enum import Enum

import matplotlib.pyplot as plt

from mockingbird.config import load_config
from mockingbird.conversation.history import History
from mockingbird.preprocessing.factory import make_processor
from mockingbird.scheduler.busy import Busy
from mockingbird.scheduler.initiation import Initiation
from mockingbird.scheduler.sleep import Sleep
from mockingbird.scheduler.weekday import Weekday


def format_time(value):
    return value.strftime("%H:%M:%S")


def format_duration(value):
    total_seconds = int(value.total_seconds())
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}"


def visualize(initiation, busy, sleep, samples=5):
    for weekday in Weekday:
        print(f"\n{'=' * 40}")
        print(weekday.name)
        print("=" * 40)

        print("\nInitiation:")
        for _ in range(samples):
            times = initiation.generate(weekday)
            print("  ", ", ".join(format_time(t) for t in times) or "None")

        print("\nBusy:")
        for _ in range(samples):
            intervals = busy.generate(weekday)
            formatted = [
                f"{format_time(start)} ({format_duration(length)})"
                for start, length in intervals
            ]
            print("  ", ", ".join(formatted) or "None")

        print("\nSleep:")
        for _ in range(samples):
            start, end = sleep.generate(weekday)
            print(f"   {format_time(start)} - {format_time(end)}")


def main():
    config = load_config()

    user = config["user"]
    other = config["other"]

    processor = make_processor(
        config["preprocessing"]["messenger"],
        None,
    )

    history: History = processor.process(
        user,
        other,
        config["paths"]["raw"],
        6000,
        12000,
    )

    # Distribution parameters
    bin_size = timedelta(minutes=15)
    conversation_window = timedelta(minutes=30)

    initiation = Initiation(
        history,
        bin_size,
        conversation_window,
    )

    busy = Busy(
        history,
        bin_size,
        conversation_window,
        max_busy_duration=timedelta(hours=3),
    )

    sleep = Sleep(
        initiation.time_distribution,
        bin_size,
        lower_percentile=0.80,
        upper_percentile=0.95,
    )

    visualize(initiation, busy, sleep, samples=1)


if __name__ == "__main__":
    main()
