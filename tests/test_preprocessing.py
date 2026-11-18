import json
from pathlib import Path

from mockingbird.preprocessing.telegram.processor import TelegramProcessor


def test_only_original_pair_messages_are_used(tmp_path: Path):
    data = {
        "messages": [
            {
                "id": 1,
                "type": "message",
                "from": "Persona",
                "date": "2026-01-01T12:00:00",
                "text": ["a", {"type": "bold", "text": "b"}],
            },
            {
                "id": 2,
                "type": "message",
                "from": "Partner",
                "date": "2026-01-01T12:01:00",
                "text": "context",
            },
            {
                "id": 3,
                "type": "message",
                "from": "Partner",
                "forwarded_from": "Persona",
                "date": "2026-01-01T12:02:00",
                "text": "forward",
            },
            {
                "id": 4,
                "type": "message",
                "from": "Stranger",
                "date": "2026-01-01T12:03:00",
                "text": "unknown",
            },
            {
                "id": 5,
                "type": "service",
                "from": "Persona",
                "date": "2026-01-01T12:04:00",
                "text": "service",
            },
        ]
    }
    (tmp_path / "result.json").write_text(json.dumps(data), encoding="utf-8")
    history = TelegramProcessor().process("Persona", "Partner", tmp_path)
    assert [m.message_id for m in history.messages] == [1, 2]
    assert history.messages[0].text == "ab"


def test_stable_ids_handle_renamed_authors_without_changing_persona_identity(
    tmp_path: Path,
):
    data = {
        "messages": [
            {
                "id": 1,
                "from": "Old nickname",
                "from_id": "user1",
                "date": "2026-11-01T12:00:00",
                "text": "own words",
            },
            {
                "id": 2,
                "from": "New partner nickname",
                "from_id": "user2",
                "date": "2026-11-01T12:01:00",
                "text": "context",
            },
            {
                "id": 3,
                "from": "Persona",
                "from_id": "stranger",
                "date": "2026-11-01T12:02:00",
                "text": "same-name stranger",
            },
        ]
    }
    (tmp_path / "result.json").write_text(json.dumps(data), encoding="utf-8")
    history = TelegramProcessor(
        author_ids={"user1": "Persona", "user2": "Partner"}
    ).process("Persona", "Partner", tmp_path)
    assert [m.author for m in history.messages] == ["Persona", "Partner"]
    assert [m.message_id for m in history.messages] == [1, 2]
