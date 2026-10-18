import argparse
import random
from pathlib import Path

from mockingbird.conversation.history import History
from mockingbird.preprocessing.telegram.processor import TelegramProcessor

PROJECT_ROOT = Path(__file__).resolve().parents[3]

PERSONALITY_SYSTEM_PROMPT = """
You are an analyst describing a person's communication style and personality from samples of their chat messages. Your description will be used to imitate this person in a chat assistant, so it must be concrete,evidence-based, and specific enough to reproduce their voice.

## Input
You will receive a numbered list of messages written by one person. Treat these as the only evidence available.

## What to describe
Cover EVERY section below. If the samples contain no evidence for a section, write "Not enough evidence" for that section instead of guessing.

### 1. Tone and register
- Formal / casual / mixed, and how consistent this is
- Warmth: distant, friendly, affectionate — with examples of the markers
- Directness: blunt vs. softened (hedges, "maybe", "kind of")
- Energy: calm, animated, clipped, scattered

### 2. Message shape
- Use of sentence fragments, one-word replies, run-ons
- Capitalization habit: standard, all-lowercase, ALL-CAPS emphasis, mixed
- Punctuation habit: heavy, minimal, missing terminal punctuation, ellipses, dashes

### 3. Humor
- Presence and style: dry, sarcastic, self-deprecating, absurd, playful, none
- Does humor appear in serious moments, or only light ones?

### 4. Emotion expression
- Range: which emotions appear (joy, frustration, sadness, anxiety, excitement, \
boredom, affection, anger)
- Intensity: understated vs. exaggerated
- How negative emotion is shown: directly, indirectly, humor, silence
- How positive emotion is shown: words, emoji, punctuation, brevity

### 5. Vocabulary
- Slang, colloquialisms, regionalisms (list examples)
- Profanity: none / mild / frequent, and in what contexts
- Favorite words, fillers, or verbal tics ("honestly", "like", "just", "tbh", "ngl")
- Abbreviations and acronyms used
- Vocabulary level: plain, technical, literary, playful

### 6. Interaction patterns
- Questions: how often, and what kind (genuine, rhetorical, deflecting)
- Agreement / disagreement: how they say yes and no
- Requests: direct imperative vs. softened
- Acknowledgment: "ok", "sure", "got it", or something idiosyncratic
- How they open and close exchanges

### 7. Topics and interests
- What subjects they return to
- What they volunteer vs. only respond to
- How much they self-disclose

### 8. Distinctive quirks
- Anything else that makes their messages recognizable: catchphrases, typing habits, deliberate misspellings, formatting patterns, greetings, sign-offs, nicknames

## Rules
- Ground every claim in the samples. Prefer "uses lowercase almost always, e.g. 'yeah i think so'" over "casual".
- Distinguish what is frequent from what happened once.
- Do not invent details.

## Output format
Return plain prose organized under the nine headings above, in order. Each section: two to five sentences, or "Not enough evidence". End with a one-paragraph "Summary voice" that a writer could use as a style guide.

The samples are as follows:
"""


def raw_prompt(history: History, max_samples: int | None = None) -> str:
    messages = history.users_messages()

    if max_samples is not None and len(messages) > max_samples:
        messages = random.sample(messages, max_samples)
        messages.sort(key=lambda message: message.timestamp)

    samples = "\n".join(f"{message.text}" for message in messages)
    return f"{PERSONALITY_SYSTEM_PROMPT}\n{samples}\n"


class Inner:
    def __init__(self, path: Path) -> None:
        self.inner_prompt = path.read_text(encoding="utf-8")

    def __str__(self) -> str:
        return self.inner_prompt

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a raw personality prompt from a Telegram chat export."
    )
    parser.add_argument("data_path", type=Path, help="Path to the exported JSON file")
    parser.add_argument("user", help="Username of the person to model")
    parser.add_argument("other", help="Username of the other participant")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed",
        help="Root output directory (default: <project>/data/processed)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Max number of user messages to include (default: all)",
    )
    args = parser.parse_args()

    processor = TelegramProcessor()
    history = processor.process(args.user, args.other, args.data_path)

    if not history.users_messages():
        raise SystemExit(f"No messages found for {args.user!r} in {args.data_path}")

    prompt = raw_prompt(history, args.max_samples)

    out_dir = args.output_root / "inner"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{args.user.replace(" ", "_")}_to_LLM.txt"
    out_file.write_text(prompt, encoding="utf-8")

    print(f"Wrote raw prompt to {out_file} ({len(prompt)} chars)")


if __name__ == "__main__":
    main()