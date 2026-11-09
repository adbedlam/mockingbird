import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from mockingbird.conversation.message import Message as ConversationMessage
from mockingbird.llm.client import LLM
from mockingbird.llm.message import Role


@dataclass
class Diary:
    path: Path | None
    user: str
    other: str
    update_every_messages: int
    model: LLM | None = None
    inner_prompt: str = ""
    entries: list[str] = field(init=False, default_factory=list)
    pending: list[dict[str, Any]] = field(init=False, default_factory=list)
    seen_fingerprints: set[str] = field(init=False, default_factory=set)
    updates: list[dict[str, str | int]] = field(init=False, default_factory=list)
    chat_turns: list[dict[str, Any]] = field(init=False, default_factory=list)

    def __post_init__(self) -> None:
        if self.path is None:
            return
        if self.update_every_messages < 1:
            raise ValueError("diary.update_every_messages must be at least 1")

        if self.path.exists():
            state = json.loads(self.path.read_text(encoding="utf-8"))
            self.entries = state.get("entries", [])
            self.pending = state.get("pending", [])
            self.seen_fingerprints = set(state.get("seen_fingerprints", []))
            self.updates = state.get("updates", [])
            self.chat_turns = state.get("chat_turns", [])
        self._write_markdown_files()

    @classmethod
    def from_config(
        cls,
        config: dict,
        user: str,
        other: str,
        model: LLM,
        inner_prompt: str,
        update_every_messages: int | None = None,
        path_root: Path | None = None,
    ) -> "Diary":
        diary_config = config.get("diary", {})
        diary_root = path_root or config["paths"]["diary"]
        path = diary_root / f"{user}_{other}".replace(" ", "_") / "diary.json"
        return cls(
            path,
            user,
            other,
            (
                update_every_messages
                if update_every_messages is not None
                else diary_config.get("update_every_messages", 100)
            ),
            model,
            inner_prompt,
        )

    @classmethod
    def disabled(cls, user: str, other: str) -> "Diary":
        return cls(None, user, other, update_every_messages=100)

    def record_messages(self, messages: list[ConversationMessage]) -> None:
        if self.path is None:
            return
        for message in messages:
            fingerprint = self._fingerprint(message)
            if fingerprint in self.seen_fingerprints:
                continue
            self.seen_fingerprints.add(fingerprint)
            self.pending.append(self._serialize(message))
        self._save()
        self._update_if_due()

    def record_incoming(self, text: str) -> None:
        timestamp = datetime.now()
        self.record_messages(
            [ConversationMessage(author=self.other, timestamp=timestamp, text=text)]
        )

    def record_chat_turn(
        self,
        incoming: str,
        response: str,
        request_messages: list[dict[str, str]] | None = None,
    ) -> None:
        if self.path is None:
            return
        self.chat_turns.append(
            {
                "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
                "incoming": incoming,
                "response": response,
                "request_messages": request_messages or [],
            }
        )
        self._save()
        self._write_markdown_files()

    def add_to_prompt(self, base_prompt: str) -> str:
        if not self.text:
            return base_prompt
        return (
            base_prompt
            + "\n\n## Long-term diary\n"
            + "Use these remembered experiences as context when relevant:\n"
            + self.text
        )

    def _update_if_due(self) -> bool:
        if len(self.pending) < self.update_every_messages:
            return False
        model = self.model
        if model is None:
            return False

        transcript = self._transcript(self.pending)
        prompt = f"""Write a short new diary passage for {self.user}, who is the modeled person.
The conversation partner is {self.other}.

Decide whether these messages add a concrete, useful personal memory. If not,
skip. Do not invent events, feelings, causes, or details. Use only the supplied
profile, existing diary, and messages. Do not write about software or an agent.

If adding a memory, write it in first person and in the modeled person's
natural, conversational voice. Keep it as a short passage without a heading or
date. Do not repeat earlier diary entries.

Return only valid JSON with this shape:
{{"decision":"append" or "skip","note":"brief evidence-based explanation","entry":"new first-person passage or empty string"}}
The note must be brief and must not reveal hidden reasoning. For skip, return an
empty entry.

PERSONALITY PROFILE:
{self.inner_prompt}

EXISTING DIARY:
{self.text or '(empty)'}

NEW MESSAGES (authors are the configured participant names):
{transcript}
"""
        raw_output = model.complete(
            [
                {
                    "role": Role.SYSTEM,
                    "content": "Assess whether the evidence supports a useful personal memory.",
                },
                {"role": Role.USER, "content": prompt},
            ]
        ).strip()
        result = self._parse_update(raw_output)
        update: dict[str, str | int] = {
            "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
            "message_count": len(self.pending),
            "prompt": prompt,
            "messages": transcript,
            "decision": result["decision"],
            "note": result["note"],
            "entry": result["entry"],
            "raw_output": raw_output,
        }
        self.updates.append(update)

        if result["decision"] in {"append", "skip"}:
            if result["decision"] == "append" and result["entry"]:
                self.entries.append(result["entry"])
            self.pending.clear()

        self._save()
        self._write_markdown_files()
        return result["decision"] == "append" and bool(result["entry"])

    @staticmethod
    def _parse_update(raw_output: str) -> dict[str, str]:
        candidate = raw_output.strip()
        if candidate.startswith("```"):
            candidate = re.sub(r"^```(?:json)?\s*|\s*```$", "", candidate)
        try:
            result = json.loads(candidate)
        except json.JSONDecodeError:
            return {
                "decision": "invalid",
                "note": "Ответ модели не соответствовал ожидаемому JSON; сообщения оставлены в очереди.",
                "entry": "",
            }
        if not isinstance(result, dict):
            return {
                "decision": "invalid",
                "note": "Ответ модели должен быть JSON-объектом; сообщения оставлены в очереди.",
                "entry": "",
            }

        decision = result.get("decision")
        entry = result.get("entry", "")
        note = result.get("note", "")
        if not isinstance(decision, str) or decision not in {"append", "skip"}:
            decision = "invalid"
        if not isinstance(entry, str) or not isinstance(note, str):
            return {
                "decision": "invalid",
                "note": "В JSON модели были поля неверного типа; сообщения оставлены в очереди.",
                "entry": "",
            }
        if decision == "append" and not entry.strip():
            decision, note = "skip", note or "Модель не вернула текст записи."
        return {"decision": decision, "note": note, "entry": entry.strip()}

    @staticmethod
    def _transcript(messages: list[dict[str, Any]]) -> str:
        return "\n".join(
            f"{message['author']}@{message['timestamp']}: "
            f"{message.get('text') or ''}"
            + (
                f" (picture: {message['picture_description']})"
                if message.get("picture_description")
                else ""
            )
            for message in messages
        )

    @property
    def text(self) -> str:
        return "\n\n".join(self.entries)

    def _save(self) -> None:
        if self.path is None:
            return
        state = {
            "entries": self.entries,
            "pending": self.pending,
            "seen_fingerprints": sorted(self.seen_fingerprints),
            "updates": self.updates,
            "chat_turns": self.chat_turns,
        }
        self._atomic_write(self.path, json.dumps(state, ensure_ascii=False, indent=2))

    def _write_markdown_files(self) -> None:
        if self.path is None:
            return
        journal_path = self.path.with_name("diary.md")
        updates_path = self.path.with_name("diary_updates.md")

        journal = f"# Diary — {self.user}\n\n"
        journal += (
            "\n\n---\n\n".join(self.entries) if self.entries else "Записей пока нет.\n"
        )
        self._atomic_write(journal_path, journal)

        audit = f"# Diary update log — {self.user} и {self.other}\n\n"
        if not self.updates:
            audit += "Обновлений пока не было.\n"
        for index, update in enumerate(self.updates, start=1):
            audit += (
                f"## Обновление {index} · {update['timestamp']}\n\n"
                f"**Сообщений в пакете:** {update['message_count']}\n\n"
                f"**Решение:** {update['decision']}\n\n"
                f"**Краткое объяснение:** {update['note']}\n\n"
                "### Полный prompt\n\n"
                f"```text\n{update['prompt']}\n```\n\n"
                "### Сообщения, переданные модели\n\n"
                f"```text\n{update['messages']}\n```\n\n"
                "### Ответ модели\n\n"
                f"```json\n{update['raw_output']}\n```\n\n"
            )
            if update["entry"]:
                audit += f"### Добавленная запись\n\n{update['entry']}\n\n"
            audit += "---\n\n"
        self._atomic_write(updates_path, audit)

        chat_path = self.path.with_name("chat_transcript.md")
        chat = f"# Preview chat — {self.user} и {self.other}\n\n"
        if not self.chat_turns:
            chat += "Диалогов пока нет.\n"
        for index, turn in enumerate(self.chat_turns, start=1):
            chat += (
                f"## Реплика {index} · {turn['timestamp']}\n\n"
                f"**{self.other}:** {turn['incoming']}\n\n"
                f"**{self.user}:** {turn['response']}\n\n"
                "### Контекст, отправленный модели\n\n"
                "```text\n"
                + "\n\n".join(
                    f"[{message['role']}]\n{message['content']}"
                    for message in turn.get("request_messages", [])
                )
                + "\n```\n\n"
                "---\n\n"
            )
        self._atomic_write(chat_path, chat)

    @staticmethod
    def _atomic_write(path: Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_suffix(path.suffix + ".tmp")
        temporary_path.write_text(content, encoding="utf-8")
        temporary_path.replace(path)

    @staticmethod
    def _serialize(message: ConversationMessage) -> dict[str, Any]:
        return {
            "author": message.author,
            "timestamp": message.timestamp.isoformat(),
            "text": message.text,
            "picture_description": message.picture_description,
            "message_id": message.message_id,
        }

    @staticmethod
    def _fingerprint(message: ConversationMessage) -> str:
        identity = {
            "message_id": message.message_id,
            "author": message.author,
            "timestamp": message.timestamp.isoformat(),
            "text": message.text,
            "picture_description": message.picture_description,
        }
        canonical = json.dumps(identity, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
