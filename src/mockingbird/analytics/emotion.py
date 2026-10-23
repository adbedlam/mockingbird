from dataclasses import dataclass


@dataclass(frozen=True)
class Emotion:
    valence: float
    arousal: float
