from collections import Counter
from enum import StrEnum

from mockingbird.conversation import message
from mockingbird.llm.model import Message, Model, Role


class Emotion(StrEnum):
    EXCITED = "excited"
    CALM = "calm"
    TENSE = "tense"
    SAD = "sad"
    NEUTRAL = "neutral"


SYSTEM_PROMPT = """
Classify the emotion of the user's message.

Respond with exactly one word from this list, lowercase, no punctuation:
excited, calm, tense, sad, neutral

Definitions:
- excited: positive, high energy (enthusiastic, eager, playful)
- calm:    positive, low energy (content, warm, relaxed)
- tense:   negative, high energy (angry, frustrated, anxious)
- sad:     negative, low energy (disappointed, tired, hurt)
- neutral: factual, logistical, or no clear emotional signal
"""


def recognize_emotion(message: message.Message, model: Model) -> Emotion:
    prompt: list[Message] = [
        {"role": Role.SYSTEM, "content": SYSTEM_PROMPT},
        {"role": Role.USER, "content": message.text},
    ]
    raw = model.complete(prompt, temperature=0.0, max_tokens=10)

    text = raw.strip().lower()

    for emotion in Emotion:
        if emotion.value in text:
            return emotion
    return Emotion.NEUTRAL


def emotion_distribution(
    messages: list[message.Message], model: Model
) -> dict[Emotion, float]:
    if not messages:
        return {emotion: 0.0 for emotion in Emotion}

    counts = Counter(recognize_emotion(message, model) for message in messages)

    return {emotion: counts[emotion] / len(messages) for emotion in Emotion}
