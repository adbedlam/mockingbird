"""A fictional day, separate from evidence-backed memories of real people."""

from __future__ import annotations

import json
import logging
from copy import deepcopy
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from mockingbird.llm.client import LLM
from mockingbird.llm.message import Message, Role
from mockingbird.persona.prompts.life import (
    LIFE_CONTEXT_INSTRUCTIONS,
    LIFE_PROMPT_BUILD_INSTRUCTIONS,
    LIFE_UPDATE_DISABLED,
    life_update_protocol,
    prepare_life_prompt,
)

logger = logging.getLogger(__name__)


@dataclass
class LifePrompt:
    content: str

    @classmethod
    def from_history(cls, model: LLM, history) -> LifePrompt:
        prompt = LIFE_PROMPT_BUILD_INSTRUCTIONS.format(
            user=history.user,
            other=history.other,
        )
        messages: list[Message] = [
            Message(role=Role.SYSTEM, content=prompt),
            Message(role=Role.USER, content=history.to_samples()),
        ]
        content = model.complete(messages)
        if not content.strip():
            raise ValueError("LLM returned an empty life prompt")
        return cls(content=content.strip())

    def to_llm_message(self) -> Message:
        return Message(role=Role.SYSTEM, content=self.content)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(self.content, encoding="utf-8")
        temporary.replace(path)

    @classmethod
    def load(cls, path: Path) -> LifePrompt:
        return cls(content=path.read_text(encoding="utf-8"))


class PersonaLife:
    def __init__(
        self,
        directory: Path,
        user: str,
        other: str,
        timezone: str,
        *,
        diary_hour: int = 23,
        life_prompt: str = "",
    ):
        if not 0 <= diary_hour <= 23:
            raise ValueError("life.diary_hour must be between 0 and 23")
        self.directory = directory
        self.user, self.other = user, other
        self.zone = ZoneInfo(timezone)
        self.diary_hour = diary_hour
        self.life_prompt = life_prompt
        self.days: dict[str, dict] = {}
        self.retry_at: datetime | None = None
        path = directory / "life.json"
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            if data.get("version") not in (1, 2) or data.get("pair") != [user, other]:
                raise ValueError("Life state belongs to another persona or version")
            if data.get("timezone") != timezone:
                raise ValueError(
                    "Life state timezone changed; use a new life directory"
                )
            for key, day in data["days"].items():
                self._validate(
                    day, date.fromisoformat(key), allow_empty=data["version"] == 2
                )
                if data["version"] == 1:
                    self._identify(day)
                    # Old markers used list positions, which change on cancellation.
                    day["announced"] = [
                        f"a{int(marker.split(':')[0]) + 1}:{marker.split(':')[1]}"
                        for marker in day.get("announced", [])
                    ]
                ids = [row.get("id") for row in day["activities"]]
                if any(not isinstance(key, str) or not key for key in ids) or len(
                    set(ids)
                ) != len(ids):
                    raise ValueError("Invalid or duplicate activity IDs")
                if any(
                    not identity.startswith("a") or not identity[1:].isdigit()
                    for identity in ids
                ):
                    raise ValueError("Invalid activity ID format")
                if (
                    type(day.get("revision")) is not int
                    or day["revision"] < 0
                    or type(day.get("next_activity_id")) is not int
                    or day["next_activity_id"]
                    <= max((int(identity[1:]) for identity in ids), default=0)
                    or not isinstance(day.get("changes"), list)
                ):
                    raise ValueError("Invalid life revision metadata")
            self.days = data["days"]
            self.retry_at = (
                datetime.fromisoformat(data["retry_at"])
                if data.get("retry_at")
                else None
            )

    def local(self, now: datetime) -> datetime:
        if now.tzinfo is None:
            raise ValueError("Life clock must be timezone-aware")
        return now.astimezone(self.zone)

    @staticmethod
    def _identify(day: dict) -> None:
        for index, row in enumerate(day["activities"], 1):
            row["id"] = f"a{index}"
        day["revision"] = 0
        day["next_activity_id"] = len(day["activities"]) + 1
        day["changes"] = []

    def needed(self, now: datetime) -> bool:
        now = self.local(now)
        return now.date().isoformat() not in self.days and (
            self.retry_at is None or now >= self.retry_at
        )

    def prepare(self, now: datetime, profile: str) -> list[Message]:
        now = self.local(now)
        yesterday = self.days.get((now.date() - timedelta(days=1)).isoformat())
        return [
            Message(
                role=Role.SYSTEM,
                content=prepare_life_prompt(
                    self.user,
                    profile,
                    self.life_prompt,
                    now.isoformat(),
                ),
            ),
            Message(
                role=Role.USER,
                content=json.dumps(
                    {
                        "date": now.date().isoformat(),
                        "previous_simulated_day": yesterday,
                    },
                    ensure_ascii=False,
                ),
            ),
        ]

    @staticmethod
    def _validate(data: dict, expected: date, *, allow_empty: bool = False) -> None:
        if not isinstance(data, dict) or data.get("date") != expected.isoformat():
            raise ValueError("Life plan date mismatch")
        rows = data.get("activities")
        if (
            not isinstance(rows, list)
            or not (0 if allow_empty else 1) <= len(rows) <= 8
        ):
            raise ValueError("Expected 1–8 simulated activities")
        previous = "00:00"
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError("Invalid activity")
            for field in ("start", "end"):
                value = row.get(field)
                if not isinstance(value, str) or len(value) != 5:
                    raise ValueError("Activity time must be HH:MM")
                if time.fromisoformat(value).strftime("%H:%M") != value:
                    raise ValueError("Invalid activity time")
            if not previous <= row["start"] < row["end"]:
                raise ValueError("Activities overlap or have invalid duration")
            previous = row["end"]
            activity = row.get("activity")
            if (
                not isinstance(activity, str)
                or not activity.strip()
                or len(activity) > 500
            ):
                raise ValueError("Invalid activity text")
        if not isinstance(data.get("reason"), str) or not data["reason"].strip():
            raise ValueError("Missing visible plan explanation")

    def apply(self, raw: str, requested_date: str, now: datetime) -> dict:
        now = self.local(now)
        if requested_date != now.date().isoformat():
            raise ValueError("Day changed while generating the life plan")
        data = json.loads(raw)
        self._validate(data, now.date())
        if requested_date in self.days:
            raise ValueError("A plan already exists for this day")
        # Only the validated schema enters world state; extra model fields are discarded.
        day = {
            "date": requested_date,
            "activities": [
                {k: row[k] for k in ("start", "end", "activity")}
                for row in data["activities"]
            ],
            "reason": data["reason"],
            "origin": "simulated",
            "created_at": now.isoformat(),
            "announced": [],
        }
        self._identify(day)
        old_retry = self.retry_at
        self.days[requested_date] = day
        self.retry_at = None
        try:
            self.save()
        except OSError:
            del self.days[requested_date]
            self.retry_at = old_retry
            raise
        return day

    def update_protocol(
        self, now: datetime, incoming: list[dict], *, allowed: bool
    ) -> str:
        day = self.days.get(self.local(now).date().isoformat())
        if not allowed or day is None or not incoming:
            return LIFE_UPDATE_DISABLED
        return life_update_protocol(day, incoming)

    def update(
        self, update: dict, now: datetime, incoming: list[dict], reply: str
    ) -> dict:
        """Validate a complete change set before one atomic world-state write."""
        candidate, record = self.prepare_update(update, now, incoming, reply)
        key = candidate["date"]
        previous = self.days[key]
        self.days[key] = candidate
        try:
            self.save()
        except OSError:
            self.days[key] = previous
            raise
        return record

    def prepare_update(
        self, update: dict, now: datetime, incoming: list[dict], reply: str
    ) -> tuple[dict, dict]:
        """Validate without changing memory or disk; used before external delivery."""
        now = self.local(now)
        key = now.date().isoformat()
        day = self.days.get(key)
        if day is None or update.get("date") != key:
            raise ValueError("Life update date mismatch")
        revision = update.get("base_revision")
        if type(revision) is not int or revision != day["revision"]:
            raise ValueError("Life update revision is stale")
        source = next(
            (row for row in incoming if row["id"] == update.get("source_id")), None
        )
        if source is None:
            raise ValueError("Life update must cite a current incoming message")
        reason, changes = update.get("reason"), update.get("changes")
        if not isinstance(reason, str) or not reason.strip() or len(reason) > 1000:
            raise ValueError("Missing or oversized life update explanation")
        if not isinstance(changes, list) or not 1 <= len(changes) <= 8:
            raise ValueError("Expected 1–8 life changes")
        candidate = deepcopy(day)
        edits = []
        touched = set()
        for change in changes:
            if not isinstance(change, dict) or change.get("op") not in (
                "replace",
                "cancel",
                "add",
            ):
                raise ValueError("Invalid life operation")
            op = change["op"]
            before = None
            if op != "add":
                identity = change.get("id")
                if not isinstance(identity, str) or identity in touched:
                    raise ValueError("Duplicate or invalid activity ID in change set")
                touched.add(identity)
                before = next(
                    (row for row in candidate["activities"] if row["id"] == identity),
                    None,
                )
                if before is None:
                    raise ValueError("Unknown activity ID")
                start = datetime.fromisoformat(f"{key}T{before['start']}").replace(
                    tzinfo=self.zone
                )
                if start <= now:
                    raise ValueError("Cannot change a completed or current activity")
                candidate["activities"].remove(before)
            else:
                if "id" in change:
                    raise ValueError("New activity IDs are assigned locally")
                identity = f"a{candidate['next_activity_id']}"
                candidate["next_activity_id"] += 1
            after = None
            if op != "cancel":
                after = {
                    "id": identity,
                    **{
                        field: change.get(field)
                        for field in ("start", "end", "activity")
                    },
                }
                # Validate types/times before sorting or comparing the entire plan.
                self._validate(
                    {"date": key, "reason": reason, "activities": [after]}, now.date()
                )
                start = datetime.fromisoformat(f"{key}T{after['start']}").replace(
                    tzinfo=self.zone
                )
                if start <= now:
                    raise ValueError("New activities must start in the future")
                if before == after:
                    raise ValueError("Life operation makes no change")
                candidate["activities"].append(after)
            edits.append({"op": op, "before": before, "after": after})
        candidate["activities"].sort(key=lambda row: row["start"])
        self._validate(candidate, now.date(), allow_empty=True)
        candidate["revision"] += 1
        record = {
            "date": key,
            "revision": candidate["revision"],
            "at": now.isoformat(),
            "reason": reason.strip(),
            "source": deepcopy(source),
            "reply": reply,
            "edits": edits,
        }
        candidate["changes"].append(record)
        return candidate, record

    @staticmethod
    def describe_change(record: dict) -> str:
        lines = [f"Plan {record['date']}, revision {record['revision']}"]
        for edit in record["edits"]:

            def label(row):
                return f"{row['start']}–{row['end']} {row['activity']}"

            if edit["op"] == "cancel":
                lines.append("Cancelled: " + label(edit["before"]))
            elif edit["op"] == "add":
                lines.append("Added: " + label(edit["after"]))
            else:
                lines.append(
                    "Replaced: " + label(edit["before"]) + " -> " + label(edit["after"])
                )
        return "\n".join(lines)

    def defer(self, now: datetime) -> None:
        self.retry_at = self.local(now) + timedelta(minutes=5)
        self.save()

    def rows(self, day: dict, now: datetime) -> list[dict]:
        now = self.local(now)
        result = []
        for row in day["activities"]:
            start = datetime.fromisoformat(f"{day['date']}T{row['start']}").replace(
                tzinfo=self.zone
            )
            end = datetime.fromisoformat(f"{day['date']}T{row['end']}").replace(
                tzinfo=self.zone
            )
            status = (
                "completed" if now >= end else "current" if now >= start else "planned"
            )
            result.append({**row, "status": status})
        return result

    def advance(
        self, now: datetime, *, checkpoint: bool = False
    ) -> list[tuple[str, str]]:
        now = self.local(now)
        previous = deepcopy(self.days)
        events: list[tuple[str, str]] = []
        changed = False
        for key, day in self.days.items():
            rows = self.rows(day, now)
            if key == now.date().isoformat():
                for row in rows:
                    marker = f"{row['id']}:{row['status']}"
                    if row["status"] != "planned" and marker not in day["announced"]:
                        day["announced"].append(marker)
                        changed = True
                        events.append(
                            (
                                "LIFE_EVENT",
                                f"Simulation | {row['start']}–{row['end']} | "
                                + (
                                    "completed"
                                    if row["status"] == "completed"
                                    else "current"
                                )
                                + f" | {row['activity']}",
                            )
                        )
            if (
                key < now.date().isoformat()
                or now.hour >= self.diary_hour
                or checkpoint
            ):
                done = [r["activity"] for r in rows if r["status"] == "completed"]
                text = (
                    "today " + "; ".join(done)
                    if done
                    else "no completed activities yet"
                )
                # One record per calendar date; checkpoints update it instead of duplicating it.
                if day.get("diary", {}).get("content") != text:
                    day["diary"] = {
                        "content": text,
                        "recorded_at": now.isoformat(),
                        "origin": "simulated",
                    }
                    changed = True
                    events.append(
                        (
                            "DAILY_DIARY",
                            f"{key} | Simulated day, diary entry saved: {text}",
                        )
                    )
        if changed:
            try:
                self.save()
            except OSError:
                self.days = previous
                raise
        return events

    def context(self, now: datetime) -> str:
        now = self.local(now)
        day = self.days.get(now.date().isoformat())
        state = self.rows(day, now) if day else []
        return LIFE_CONTEXT_INSTRUCTIONS + json.dumps(
            {
                "at": now.isoformat(),
                "date": day["date"] if day else None,
                "revision": day["revision"] if day else None,
                "activities": state,
                "recent_plan_changes": day["changes"][-3:] if day else [],
                "recent_simulated_diary": [
                    {"date": key, **day["diary"]}
                    for key, day in sorted(self.days.items())
                    if "diary" in day
                    and now.date() - timedelta(days=7)
                    <= date.fromisoformat(key)
                    < now.date()
                ],
            },
            ensure_ascii=False,
        )

    def view(self, now: datetime) -> str:
        day = self.days.get(self.local(now).date().isoformat())
        if day is None:
            return "The simulated-day plan has not been created yet"
        labels = {"completed": "completed", "current": "current", "planned": "planned"}
        return (
            f"Simulated day {day['date']} — {self.user}, revision {day['revision']}\n"
            + "\n".join(
                f"{row['id']} | {row['start']}–{row['end']} | {labels[row['status']]} | {row['activity']}"
                for row in self.rows(day, now)
            )
            + f"\nChoice reason: {day['reason']}"
            + (
                "\nLatest change:\n" + self.describe_change(day["changes"][-1])
                if day["changes"]
                else ""
            )
            + f"\nDiary: {self.directory / 'daily_diary.md'}"
        )

    def diary_view(self) -> str:
        return (
            "\n\n".join(
                f"{key} | [SIMULATION]\n{day['diary']['content']}"
                for key, day in sorted(self.days.items())
                if "diary" in day
            )
            or "No simulated-day diary entries yet"
        )

    def save(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        target = self.directory / "life.json"
        temporary = target.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(
                {
                    "version": 2,
                    "pair": [self.user, self.other],
                    "timezone": self.zone.key,
                    "retry_at": self.retry_at.isoformat() if self.retry_at else None,
                    "days": self.days,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        temporary.replace(target)
        try:
            (self.directory / "daily_diary.md").write_text(
                "# Simulated-day diary\n\nThese are fictional role-play events, not facts from Telegram.\n\n"
                + self.diary_view(),
                encoding="utf-8",
            )
        except OSError:
            logger.warning("Life JSON saved, but readable diary could not be rendered")
        try:
            records = [
                record
                for _, day in sorted(self.days.items())
                for record in day["changes"]
            ]
            (self.directory / "plan_changes.md").write_text(
                "# Simulated-plan changes\n\n"
                + "\n\n".join(
                    f"## {record['at']}\n\n{self.describe_change(record)}\n\n"
                    f"Reason: {record['reason']}\n\n"
                    f"Incoming: {record['source']['text']}\n\nReply: {record['reply']}"
                    for record in records
                ),
                encoding="utf-8",
            )
        except OSError:
            logger.warning(
                "Life JSON saved, but readable change log could not be rendered"
            )
