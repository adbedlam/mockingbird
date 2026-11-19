import json
from pathlib import Path

from mockingbird.conversation.history import History


def select_persona_history(history: History, config: dict) -> History:
    settings = config.get("persona_build", {})
    selected = History.latest_history(
        history,
        settings.get("recent_messages", 1000),
        config.get("scheduler", {}).get("timezone", "Europe/Moscow"),
    )
    max_chars = settings.get("max_input_chars", 160000)
    if max_chars < 1:
        raise ValueError("persona_build.max_input_chars must be positive")
    # Preserve complete messages; never cut an author's statement midway.
    total = 0
    recent = []
    for message in reversed(selected.messages):
        size = len(message.to_sample()) + 1
        if total + size > max_chars:
            break
        recent.append(message)
        total += size
    if not recent:
        raise ValueError("No recent source messages fit the persona build budget")
    return History(history.user, history.other, list(reversed(recent)))


def load_manifest(processed: Path) -> dict:
    path = processed / "persona_sources.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def save_manifest(processed: Path, data: dict) -> None:
    processed.mkdir(parents=True, exist_ok=True)
    target = processed / "persona_sources.json"
    temporary = target.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(target)
