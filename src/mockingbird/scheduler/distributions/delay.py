from mockingbird.scheduler.distributions.distribution import (
    Distribution,
    smooth_distribution,
)


def delay_distribution(delays: list[float], sigma: float) -> Distribution[float]:
    """One-second grid: sigma is seconds, rather than distinct-value positions."""
    if not delays:
        raise ValueError("No delays found in history")
    if any(delay < 0 or delay > 86400 for delay in delays):
        raise ValueError("Delays must be between zero and 24 hours")
    counts = [0.0] * (int(max(delays)) + 1)
    for delay in delays:
        counts[int(delay)] += 1
    return smooth_distribution(
        Distribution([float(i) for i in range(len(counts))], counts), sigma=sigma
    )
