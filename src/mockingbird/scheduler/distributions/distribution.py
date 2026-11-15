from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Generic, TypeVar
import math

ValueType = TypeVar("ValueType")


@dataclass
class Distribution(Generic[ValueType]):
    values: list[ValueType]
    probabilities: list[float]

    def sample(self) -> ValueType:
        return random.choices(
            self.values,
            weights=self.probabilities,
            k=1,
        )[0]

def smooth_distribution(
    distribution: Distribution[ValueType],
    sigma: float = 0.5,
    circular: bool = False,
) -> Distribution[ValueType]:
    if sigma <= 0:
        raise ValueError("sigma must be positive")

    if len(distribution.values) != len(distribution.probabilities):
        raise ValueError("Values and probabilities must have equal lengths")

    if not distribution.values:
        raise ValueError("Cannot smooth an empty distribution")

    if any(p < 0 for p in distribution.probabilities):
        raise ValueError("Probabilities cannot be negative")

    if not any(distribution.probabilities):
        raise ValueError("Distribution has zero total probability")

    radius = math.ceil(3 * sigma)

    kernel = {
        offset: math.exp(-(offset**2) / (2 * sigma**2))
        for offset in range(-radius, radius + 1)
    }

    smoothed = [0.0] * len(distribution.values)

    for index, probability in enumerate(distribution.probabilities):
        for offset, weight in kernel.items():
            target = index + offset

            if circular:
                target %= len(smoothed)
            else:
                target = max(0, min(target, len(smoothed) - 1))

            smoothed[target] += probability * weight

    total = sum(smoothed)
    smoothed = [p / total for p in smoothed]

    return Distribution(
        values=distribution.values.copy(),
        probabilities=smoothed,
    )