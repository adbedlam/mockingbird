import json
from types import SimpleNamespace

import pytest

from mockingbird.llm.client import LLM
from mockingbird.llm.message import Message, Role
from mockingbird.mockingbird.draft import parse_draft
from mockingbird.scheduler.scheduler import MessageKind


def test_plain_reply_is_usable_without_inventing_model_reason():
    text = "Пока никто не писал, собес в среду. Завод отказал, сказали полиграф не прошёл \U0001f643"
    draft = parse_draft(text, MessageKind.REPLY)
    assert draft.send
    assert draft.text == text
    assert draft.reason == ""
    assert draft.format == "plain_reply"


@pytest.mark.parametrize("kind", [MessageKind.INITIATIVE, None])
def test_plain_text_never_implies_an_initiative_decision(kind):
    with pytest.raises(ValueError):
        parse_draft("Привет, есть минутка?", kind)


@pytest.mark.parametrize(
    "raw",
    [
        '{"send": true,',
        '[{"text": "hi"}',
        '```json\n{"send": true,\n```',
        'Here is the result: {"send": false, "text": "", "reason": "no topic"}',
        '"unfinished string',
        "",
        "null",
    ],
)
def test_invalid_structured_output_is_not_sent_as_chat_text(raw):
    with pytest.raises(ValueError):
        parse_draft(raw, MessageKind.REPLY)


def test_valid_json_fences_and_bom_are_supported():
    raw = (
        "\ufeff```JSON\n"
        + json.dumps({"send": True, "text": "Привет", "reason": "Отвечаю"})
        + "\n```"
    )
    draft = parse_draft(raw, MessageKind.REPLY)
    assert draft.format == "json"
    assert draft.reason == "Отвечаю"


def test_plain_code_reply_keeps_its_code_fences():
    raw = "```python\nprint(1)\n```"
    draft = parse_draft(raw, MessageKind.REPLY)
    assert draft.text == raw
    assert draft.format == "plain_reply"


@pytest.mark.parametrize("json_mode", [False, True])
def test_json_mode_is_requested_at_the_api_layer_only_when_enabled(
    monkeypatch, json_mode
):
    model = LLM("http://127.0.0.1:1234/v1", "test-only", json_mode=json_mode)
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            model="test-only",
            usage=None,
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content='{"ok": true}'),
                    finish_reason="stop",
                )
            ],
        )

    monkeypatch.setattr(model.client.chat.completions, "create", create)
    assert (
        model.complete([Message(role=Role.USER, content="Return JSON")])
        == '{"ok": true}'
    )
    if json_mode:
        assert calls[0]["response_format"] == {"type": "json_object"}
    else:
        assert "response_format" not in calls[0]
