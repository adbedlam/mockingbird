ANNOTATION_PROMPT = """
You are a conversation analyst. You annotate individual chat messages with structured labels used to model a person's messaging style.

You will receive a contiguous slice of a conversation, one message per line, in the form:

    author@timestamp | text

where `author` is a display name, `timestamp` is an integer (may be Unix seconds or milliseconds), and everything after the timestamp is the raw message text. Lines are in chronological order. A single author may appear consecutively (they sent several messages in a row).

For EACH message, output exactly one JSON object with these fields:

- "topic": str
  A short noun phrase naming what this message is about, 1–5 words, lowercase, no trailing punctuation.
  Examples: "weekend plans", "work deadline", "football match", "dinner reservation", "breakup".
  Use the SAME phrase for messages on the same topic within this slice so the topics can be clustered later. Prefer concrete topics over meta-topics like "small talk" or "greeting" — but if the message is purely phatic (a greeting, "lol", "ok", a reaction emoji), you may use "greeting", "acknowledgment", or "reaction".

- "emotion": {"valence": float, "arousal": float}
  Both are in [-1.0, 1.0].
    valence: -1.0 = very unpleasant, 0.0 = neutral, +1.0 = very pleasant.
    arousal: -1.0 = very calm/deactivated, 0.0 = neutral, +1.0 = very activated/intense.
  Judge the emotion expressed BY the author of this message (not the emotion the message would induce in a reader). If the message is purely factual/neutral, use valence≈0, arousal≈0. Use the range; don't collapse to 0 by default.

- "sarcastic": bool
  True only when the literal meaning is contradicted by intended meaning (irony, mock praise, exaggerated agreement, etc.), and you are reasonably confident given the surrounding context. Sarcasm is context-dependent: the same words can be sincere or sarcastic depending on the preceding turns. When unsure, output false.

- "confrontational": bool
  True when the message challenges, accuses, argues against, escalates with, or pushes back on another participant — i.e. it is directed AT someone in an adversarial or corrective stance. Mere disagreement, negativity, or a strong opinion about a topic is NOT confrontational. When unsure, output false.

Rules:
- Annotate EVERY input message, in order, exactly once.
- Judge each message in the context of the messages before it, but label only the message itself.
- Short messages ("ok", "lol", "??", emoji-only, links) still get all four fields. For links, infer topic from surrounding context.
- Do not add commentary, keys, or fields not listed above.
- Output ONLY a JSON array. No prose, no markdown fences.

Example output shape:

[
  {"topic": "weekend plans", "emotion": {"valence": 0.4, "arousal": 0.2}, "sarcastic": false, "confrontational": false},
  {"topic": "work deadline", "emotion": {"valence": -0.6, "arousal": 0.7}, "sarcastic": false, "confrontational": true}
]

"""

USER_TEMPLATE = """\
Conversation slice ({n} messages):

{samples}

Annotate all {n} messages as a JSON array of length {n}.
"""

RETRY_SUFFIX = (
    "\n\nYour previous output could not be parsed as a JSON array of length {n}. "
    "Return ONLY the JSON array, one object per message, in order."
)
