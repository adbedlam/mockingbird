"""Local lexical memory retrieval. No embeddings, provider calls or learned facts.

Search uses current revisions, light word normalization and BM25. A small
recent-memory reserve keeps conversational continuity; a character budget bounds
the selected records. Similar meaning without shared words can still be missed.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from mockingbird.persona.diary.entry import Entry

_WORDS = re.compile(r"[a-zа-яё0-9]+", re.IGNORECASE)
_STOP = frozenset(
    "я ты он она они мы вы мне меня тебе тебя его ее их мой твой наш ваш это этот эта эти вот тут там что как где когда почему зачем ли же бы был была были быть есть будет будут да нет не ни и или а но в во на у к ко с со за по из от до об о про для при без уже еще пока теперь сегодня вчера завтра помнишь помню расскажи скажи пожалуйста дела делал делала делать план планы i me my you your he she it they we our the a an and or but to of in on at for with from is are was were be been do does did about what how when where please still now then told said".split()
)
_RU_ENDINGS = tuple(
    sorted(
        (
            "ирование",
            "ирования",
            "ировать",
            "ировали",
            "ировал",
            "ировала",
            "ировать",
            "ировать",
            "ениями",
            "ением",
            "ениями",
            "ения",
            "ение",
            "ания",
            "ание",
            "иями",
            "ями",
            "ами",
            "лась",
            "лись",
            "лось",
            "ать",
            "ять",
            "ить",
            "ала",
            "али",
            "ало",
            "или",
            "ила",
            "ило",
            "ные",
            "ного",
            "ной",
            "ный",
            "ным",
            "ных",
            "ого",
            "ему",
            "ому",
            "ыми",
            "ими",
            "его",
            "ать",
            "ить",
            "ал",
            "ил",
            "ов",
            "ев",
            "ам",
            "ям",
            "ах",
            "ях",
            "ом",
            "ем",
            "ой",
            "ей",
            "ый",
            "ий",
            "ая",
            "ое",
            "ые",
            "а",
            "я",
            "ы",
            "и",
            "у",
            "ю",
            "е",
            "о",
            "ь",
        ),
        key=len,
        reverse=True,
    )
)


def _stem(word: str) -> str:
    word = word.casefold().replace("ё", "е")
    if re.fullmatch(r"[а-я]+", word):
        for _ in range(2):
            for ending in _RU_ENDINGS:
                if word.endswith(ending) and len(word) - len(ending) >= 3:
                    word = word[: -len(ending)]
                    break
            else:
                break
        return word
    elif word.isascii() and word.isalpha():
        if word.endswith("ies") and len(word) > 4:
            return word[:-3] + "y"
        for ending in ("ing", "ed", "s"):
            if word.endswith(ending) and len(word) - len(ending) >= 3:
                return word[: -len(ending)]
    return word


@dataclass(frozen=True)
class MemorySearchConfig:
    enabled: bool = True
    recent_entries: int = 5
    max_chars: int = 24000
    min_score: float = 0.2
    max_age_days: int | None = None
    aliases: dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        if type(self.enabled) is not bool:
            raise ValueError("memory_search.enabled must be boolean")
        if type(self.recent_entries) is not int or self.recent_entries < 0:
            raise ValueError("memory_search.recent_entries must be nonnegative")
        if type(self.max_chars) is not int or self.max_chars < 1:
            raise ValueError("memory_search.max_chars must be positive")
        if (
            type(self.min_score) not in (int, float)
            or not math.isfinite(self.min_score)
            or self.min_score <= 0
        ):
            raise ValueError("memory_search.min_score must be positive and finite")
        if self.max_age_days is not None and (
            type(self.max_age_days) is not int or self.max_age_days < 1
        ):
            raise ValueError("memory_search.max_age_days must be null or positive")
        if not isinstance(self.aliases, dict) or any(
            not isinstance(k, str)
            or not isinstance(v, str)
            or _WORDS.fullmatch(k) is None
            or _WORDS.fullmatch(v) is None
            for k, v in self.aliases.items()
        ):
            raise ValueError(
                "memory_search.aliases must map single words to single words"
            )

    @classmethod
    def from_dict(cls, data: dict | None = None) -> MemorySearchConfig:
        if data is None:
            return cls()
        if not isinstance(data, dict) or any(not isinstance(key, str) for key in data):
            raise ValueError("memory_search must be a mapping")
        unknown = set(data) - {
            "enabled",
            "recent_entries",
            "max_chars",
            "min_score",
            "max_age_days",
            "aliases",
        }
        if unknown:
            raise ValueError(
                "Unknown memory_search settings: " + ", ".join(sorted(unknown))
            )
        return cls(**data)


@dataclass(frozen=True)
class MemoryHit:
    entry: Entry
    score: float
    matched_terms: tuple[str, ...]
    reason: str
    historical: bool


@dataclass(frozen=True)
class MemorySelection:
    query: str
    hits: tuple[MemoryHit, ...]
    considered: int
    eligible: int
    chars: int
    max_chars: int
    enabled: bool
    oversized: int = 0
    context_query: str = ""

    @property
    def entries(self) -> list[Entry]:
        return [hit.entry for hit in self.hits]

    def report(self) -> dict:
        return {
            "query": self.query,
            "context_query": self.context_query,
            "considered": self.considered,
            "eligible": self.eligible,
            "chars": self.chars,
            "max_chars": self.max_chars,
            "enabled": self.enabled,
            "oversized": self.oversized,
            "selected": [
                {
                    "memory_id": hit.entry.memory_id,
                    "revision": hit.entry.revision,
                    "timestamp": hit.entry.timestamp.isoformat(),
                    "state": hit.entry.state,
                    "subject": hit.entry.subject,
                    "topic": hit.entry.topic,
                    "score": round(hit.score, 5),
                    "matched_terms": list(hit.matched_terms),
                    "reason": hit.reason,
                    "historical": hit.historical,
                }
                for hit in self.hits
            ],
        }


def select_memories(
    entries: list[Entry],
    query: str,
    now: datetime,
    *,
    timezone: str,
    limit: int = 30,
    recent_days: int = 365,
    config: MemorySearchConfig | None = None,
    user: str = "",
    other: str = "",
    context_query: str = "",
) -> MemorySelection:
    if limit < 1 or recent_days < 1 or now.tzinfo is None:
        raise ValueError("Memory selection requires positive limits and an aware clock")
    cfg = config or MemorySearchConfig()
    zone = ZoneInfo(timezone)

    def local(at: datetime) -> datetime:
        return at.replace(tzinfo=zone) if at.tzinfo is None else at.astimezone(zone)

    now = local(now)
    query = query[-4000:]
    context_query = context_query[-2000:]
    aliases = {_stem(k): _stem(v) for k, v in cfg.aliases.items()}

    def tokens(text: str) -> list[str]:
        words = (w.casefold().replace("ё", "е") for w in _WORDS.findall(text))
        return [
            aliases.get(_stem(w), _stem(w))
            for w in words
            if w not in _STOP and (len(w) > 1 or w.isdigit())
        ]

    current: dict[str, Entry] = {}
    for row in entries:
        if row.memory_id is None:
            digest = hashlib.sha256(
                json.dumps(
                    [row.timestamp.isoformat(), row.content, row.source_ids],
                    ensure_ascii=False,
                ).encode()
            ).hexdigest()[:20]
            row = replace(row, memory_id="r" + digest)
        key = str(row.memory_id)
        old = current.get(key)
        if old is None or (row.revision, local(row.timestamp)) > (
            old.revision,
            local(old.timestamp),
        ):
            current[key] = row
    earliest = (
        now - timedelta(days=cfg.max_age_days) if cfg.max_age_days is not None else None
    )
    eligible = [
        e
        for e in current.values()
        if local(e.timestamp) <= now
        and (earliest is None or local(e.timestamp) >= earliest)
    ]
    recent = sorted(
        (
            e
            for e in eligible
            if local(e.timestamp) >= now - timedelta(days=recent_days)
        ),
        key=lambda e: (local(e.timestamp), str(e.memory_id)),
        reverse=True,
    )
    owners = {
        "persona": user,
        "partner": other,
        "shared": user + " " + other,
        "unknown": "",
    }
    documents: dict[str, Counter[str]] = {}
    for entry in eligible:
        terms = Counter(tokens(entry.content))
        terms.update(tokens(entry.topic) * 2)
        terms.update(tokens(owners.get(entry.subject, "")))
        terms[str(local(entry.timestamp).year)] += 1
        documents[str(entry.memory_id)] = terms
    frequencies = Counter(word for terms in documents.values() for word in terms)
    average_length = sum(sum(terms.values()) for terms in documents.values()) / max(
        1, len(documents)
    )
    query_terms = set(tokens(query)) if cfg.enabled else set()
    context_terms = set(tokens(context_query)) if cfg.enabled else set()
    if not query_terms:
        query_terms, context_terms = context_terms, set()
    scores: dict[str, float] = {}
    matched: dict[str, tuple[str, ...]] = {}
    for key, terms in documents.items():
        primary = query_terms & terms.keys()
        common = (
            tuple(sorted((query_terms | context_terms) & terms.keys()))
            if primary
            else ()
        )
        score = 0.0
        for word in common:
            inverse_frequency = math.log(
                1
                + (len(documents) - frequencies[word] + 0.5) / (frequencies[word] + 0.5)
            )
            count = terms[word]
            denominator = count + 1.2 * (
                0.25 + 0.75 * sum(terms.values()) / max(1, average_length)
            )
            weight = 1.0 if word in query_terms else 0.25
            score += weight * inverse_frequency * count * 2.2 / denominator
        scores[key], matched[key] = score, common
    ranked = sorted(
        (e for e in eligible if scores[str(e.memory_id)] >= cfg.min_score),
        key=lambda e: (scores[str(e.memory_id)], local(e.timestamp), str(e.memory_id)),
        reverse=True,
    )
    hits: dict[str, MemoryHit] = {}
    size = 0
    oversized: set[str] = set()

    def take(entry: Entry, reason: str) -> None:
        nonlocal size
        key = str(entry.memory_id)
        if key in hits or len(hits) >= limit:
            return
        cost = len(
            json.dumps(
                asdict(entry),
                ensure_ascii=False,
                default=lambda value: value.isoformat(),
            )
        )
        if size + cost > cfg.max_chars:
            oversized.add(key)
            return
        size += cost
        hits[key] = MemoryHit(
            entry,
            scores[key],
            matched[key],
            reason,
            local(entry.timestamp) < now - timedelta(days=recent_days),
        )

    reserve = min(cfg.recent_entries, limit // 3) if cfg.enabled else 0
    for entry in recent[:reserve]:
        take(entry, "recent")
    for entry in ranked:
        take(entry, "relevance")
    for entry in recent:
        take(entry, "recent")
    ordered = tuple(
        sorted(
            hits.values(),
            key=lambda hit: (local(hit.entry.timestamp), str(hit.entry.memory_id)),
        )
    )
    return MemorySelection(
        query,
        ordered,
        len(current),
        len(eligible),
        size,
        cfg.max_chars,
        cfg.enabled,
        len(oversized),
        context_query,
    )
