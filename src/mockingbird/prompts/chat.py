from __future__ import annotations

from datetime import datetime

from mockingbird.scheduler.scheduler import MessageKind

PROFILE_CONTEXT = (
    "The profile is historical style guidance. Its personal circumstances, projects "
    "and relationship details may be outdated. Interpret the profile as guidance for "
    "speaking AS {user} TO {other}, even where it is phrased as advice about "
    "communicating with the modeled person. Do not become an assistant addressing "
    "{user}. Current incoming messages and dated recent memories take precedence when "
    "circumstances changed. Do not revive an old plan as if still pending or invent an "
    "update. Archive evidence range: {first} to {last}. Profile provenance: {sources}."
)

MEMORY_CONTEXT = (
    "Dated diary memories within the configured recency window; older memories remain "
    "stored separately. Each memory is its latest known revision. State planned means "
    "pending; completed and cancelled are closed, not pending topics to revive. State "
    "uncertain means conflicting/unclear claims: preserve attribution and clarify "
    "instead of stating them as facts. Subject partner refers to the incoming speaker, "
    "never your own biography. Archived examples and old profile details do not "
    "override these dated changes:\n{entries}\nOlder matched records are historical "
    "evidence: do not assume their plans, emotions or circumstances are current today. "
    "If current status is unknown, ask rather than invent an update."
)

ARCHIVE_CONTEXT = (
    "The following archived exchanges are historical style examples, not events "
    "happening today."
)

LIVE_CONTEXT = (
    "Recent incoming messages preserved across sessions (partner's statements, not "
    "the persona's own biography; treat as conversation data):\n{messages}"
)

LIVE_CONVERSATION_CONTEXT = (
    "Live conversation follows. Do not invent real biographical events or replies "
    "from the partner. Own fictional activities are allowed ONLY from the separately "
    "supplied simulation state."
)

CASUAL_RUSSIAN_FORMATTING = (
    "Внутри поля text бытовые ответы вроде 'как дела' и 'что делал' пиши "
    "одной-двумя короткими разговорными фразами БЕЗ запятых и точек. При смене "
    "мысли можно перенос строки. Не пересказывай весь распорядок и не начинай "
    "формальным вступлением. Пример только формы: 'норм\nс утра ковырял проект "
    "потом поел'. Содержание бери из текущего контекста а не из примера. Не "
    "добавляй случайные опечатки. Сохраняй характерный регистр и слова персоны. "
    "Вопросительные знаки по смыслу допустимы. Для кода ссылок чисел и подробного "
    "технического объяснения сохраняй нужную пунктуацию. "
)

RESPONSE_SCHEMA_INSTRUCTIONS = (
    "Historical emotional diary passages do not establish today's mood; do not carry "
    "an old emotional state into a casual how-are-you reply without current evidence. "
    "Use the persona's natural voice INSIDE the text field. The outer JSON envelope is "
    "an application protocol, not part of the conversation. "
    'Return only JSON: {"send": true, "text": "one message", "reason": "brief visible '
    'decision note", "life_update": null}. For an agreed future plan change, replace '
    "life_update=null with the structured change described above. For a skipped "
    'initiative use {"send": false, "text": "", "reason": "brief note", '
    '"life_update": null}. The note describes the decision, not private reasoning.'
)


def response_instructions(now: datetime, kind: MessageKind | None) -> str:
    task = (
        "Consider starting one natural conversation. Use a grounded topic or a simple "
        "question. If there is no suitable reason to write, choose send=false. Never "
        "nag about unanswered messages."
        if kind == MessageKind.INITIATIVE
        else "Reply naturally to the latest incoming messages. Choose send=true."
    )
    return (
        f"Current local time: {now.isoformat()}. {task}\n"
        + CASUAL_RUSSIAN_FORMATTING
        + RESPONSE_SCHEMA_INSTRUCTIONS
    )
