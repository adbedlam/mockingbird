from datetime import timedelta

from mockingbird.config import load_config
from mockingbird.conversation.history import History
from mockingbird.preprocessing.factory import make_processor
from mockingbird.temporal.burst import Burst
from mockingbird.temporal.response_delay import ResponseDelay


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
        1000,
        12000,
    )

    response_delay = ResponseDelay(
        history,
        response_window=timedelta(minutes=10),
        bin_size=timedelta(seconds=1),
        smoothing_sigma=timedelta(seconds=2),
    )

    print(f"Extracted {len(response_delay.delays)} response delays")

    print("\nRaw delays:")
    for delay in response_delay.delays:
        print(f"  {delay}")

    print("\nStatistics:")
    if response_delay.delays:
        seconds = [d.total_seconds() for d in response_delay.delays]

        print(f"  Min:    {timedelta(seconds=min(seconds))}")
        print(f"  Max:    {timedelta(seconds=max(seconds))}")
        print(f"  Mean:   {timedelta(seconds=sum(seconds) / len(seconds))}")
        print(f"  Median: {timedelta(seconds=sorted(seconds)[len(seconds) // 2])}")

    print("\nSmoothed distribution:")
    for value, probability in zip(
        response_delay.distribution.values,
        response_delay.distribution.probabilities,
    ):
        if probability > 0.0001:
            print(f"  {value}: {probability:.6f}")


if __name__ == "__main__":
    main()
