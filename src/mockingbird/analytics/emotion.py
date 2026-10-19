import math
import re
from dataclasses import dataclass

from mockingbird.conversation import message as conversation
from mockingbird.llm import message as llm
from mockingbird.llm.client import LLM
from mockingbird.llm.message import Role


@dataclass(frozen=True)
class Emotion:
    valence: float
    arousal: float


NEUTRAL = Emotion(0.0, 0.0)

_ANGLES: dict[str, float] = {
    "happy": 15.0,
    "excited": 45.0,
    "alert": 75.0,
    "tense": 105.0,
    "angry": 135.0,
    "distressed": 165.0,
    "sad": 195.0,
    "depressed": 225.0,
    "bored": 255.0,
    "calm": 285.0,
    "relaxed": 315.0,
    "content": 345.0,
}


def circumplex(emotion: str, intensity: float = 1.0) -> Emotion:
    if emotion == "neutral" or emotion not in _ANGLES:
        return NEUTRAL
    rad = math.radians(_ANGLES[emotion])
    i = max(0.0, min(1.0, intensity))
    return Emotion(i * math.cos(rad), i * math.sin(rad))


SYSTEM_PROMPT = """
Classify the emotional state of a chat message on Russell's
circumplex. Reply with exactly one word from this list:

  joy-side, high arousal:  alert, excited, happy
  anger side, high arousal: tense, angry, distressed
  low arousal, negative:    sad, depressed, bored
  low arousal, positive:    calm, relaxed, content
  no clear signal:          neutral

Reply with only the label, lowercase, no punctuation.

Mild emotion still counts — do not default to neutral just because the
signal is weak. Use neutral only for pure logistics, links, numbers, and
bare acknowledgments.
"""


_VALID = set(_ANGLES) | {"neutral"}


def _parse_emotion(raw: str) -> Emotion:
    tokens = re.findall(r"[a-z]+|[\d.]+", raw.lower())
    if not tokens:
        return NEUTRAL
    label = tokens[0]
    intensity = 1.0
    if len(tokens) > 1:
        try:
            intensity = float(tokens[1])
        except ValueError:
            pass
    return circumplex(label, intensity)


def recognize_emotion(message: conversation.Message, model: LLM) -> Emotion:
    prompt: list[llm.Message] = [
        {"role": Role.SYSTEM, "content": SYSTEM_PROMPT},
        {"role": Role.USER, "content": message.text},
    ]
    raw = model.complete(prompt, temperature=0.0, max_tokens=8)
    return _parse_emotion(raw)
