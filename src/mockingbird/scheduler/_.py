from datetime import timedelta

from mockingbird.config import load_config
from mockingbird.conversation.history import History
from mockingbird.preprocessing.factory import make_processor
from mockingbird.scheduler.burst import Burst
from mockingbird.scheduler.response_delay import ResponseDelay


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

    # print("\nRaw delays:")
    # for delay in response_delay.delays:
    #     print(f"  {delay}")

    print("\nStatistics:")
    if response_delay.delays:
        seconds = [d.total_seconds() for d in response_delay.delays]

        print(f"  Min:    {timedelta(seconds=min(seconds))}")
        print(f"  Max:    {timedelta(seconds=max(seconds))}")
        print(f"  Mean:   {timedelta(seconds=sum(seconds) / len(seconds))}")
        print(f"  Median: {timedelta(seconds=sorted(seconds)[len(seconds) // 2])}")

    import matplotlib.pyplot as plt
    print("\nPlotting distribution:")

    if response_delay.delays:
        values = [
            value.total_seconds()
            for value in response_delay.distribution.values
        ]
        probabilities = response_delay.distribution.probabilities

        plt.figure(figsize=(12, 6))

        plt.bar(
            values,
            probabilities,
            width=1,
            alpha=0.7,
            label="Smoothed distribution",
        )

        plt.hist(
            [d.total_seconds() for d in response_delay.delays],
            bins=50,
            density=True,
            alpha=0.3,
            label="Raw observations",
        )

        plt.xlabel("Response delay (seconds)")
        plt.ylabel("Probability density")
        plt.title("Response Delay Distribution")
        plt.legend()
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()
