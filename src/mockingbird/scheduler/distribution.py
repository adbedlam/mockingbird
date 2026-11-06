from dataclasses import dataclass
from typing import Any


@dataclass
class Distribution:
    probabilities: list[float]
    values: list[Any]
