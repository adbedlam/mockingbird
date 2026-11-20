"""Prompt templates and instructions for the simulated persona life."""

import json

LIFE_UPDATE_DISABLED = (
    "External JSON: life_update=null. This request cannot change the plan."
)

LIFE_PROMPT_BUILD_INSTRUCTIONS = (
    "Task: Build a Life Prompt for a fictional simulated day.\n"
    "Analyze the supplied timestamped chat history for {user} and {other}. "
    "Write an instruction document that a future model can use when inventing "
    "ordinary daily activities for {user}. The output is not a plan and must "
    "not name a real or simulated date.\n\n"
    "Use {user}'s messages as the main evidence. Use {other}'s messages only "
    "to understand shared context and avoid inventing habits for {user}. "
    "Derive recurring interests, preferred activity types, realistic routines, "
    "work/rest patterns, language, boundaries, and likely sources of enjoyment "
    "or tension. Distinguish repeated patterns from isolated statements.\n\n"
    "Never invent people, meetings, jobs, diagnoses, purchases, projects, "
    "plans, relationships, or completed events that are not supported by the "
    "history. Do not turn partner requests or current concerns into facts about "
    "{user}. Do not describe real future obligations as guaranteed. Keep the "
    "life fictional and permitted by the role-play context.\n\n"
    "Write concise, information-dense instructions in English. Include "
    "negative guidance about which kinds of events to avoid. Say which details "
    "must remain speculative and which recurrent themes may guide ordinary "
    "activities. Do not reproduce every source message. Return only the Life "
    "Prompt text, without JSON or a diary-style prose essay."
)


def prepare_life_prompt(
    user: str,
    profile: str,
    life_prompt: str,
    now_iso: str,
) -> str:
    return (
        f"You are planning the FICTIONAL simulated day of persona {user}. "
        "This is a permitted role-play life, not information about a real person. "
        "Use the separate Life Prompt below as the source of everyday preferences "
        "and persona boundaries. The conversation evidence is already represented "
        "in that Life Prompt; do not repeat it as biography.\n"
        "Life Prompt:\n"
        f"{life_prompt or 'No Life Prompt is saved for this run; use only the general boundaries and profile.'}\n"
        "Invent 3–6 ordinary, concrete activities for the day, based on the persona's "
        "interests and manner. Do not reduce the day to 'sat around and did nothing'. "
        "Household tasks, walking, rest, and work on a small personal project are valid. "
        "Do not invent actions by the partner, meetings with real people, interview "
        "results, illnesses, purchases, or new biographical facts. Do not present old "
        "plans from the profile as current obligations. The plan covers today, including "
        "the already elapsed morning: this is retrospective simulation, not observation. "
        "Activities must not overlap or cross midnight. Describe each activity as a short "
        "first-person phrase in past tense, conversational and without a trailing period, "
        "for example 'sorted project files'. Do not write instructions in descriptions. "
        "Return only JSON with date (YYYY-MM-DD), activities (objects containing start, "
        "end in HH:MM, and activity), and reason (a short visible explanation of the "
        "choice, not hidden reasoning). "
        f"Exact local time: {now_iso}.\nProfile as interest guidance:\n{profile}"
    )


def life_update_protocol(day: dict, incoming: list[dict]) -> str:
    example = {
        "date": day["date"],
        "base_revision": day["revision"],
        "source_id": incoming[-1]["id"],
        "reason": "short explanation of the agreement",
        "changes": [
            {
                "op": "replace",
                "id": "ID of a future activity",
                "start": "HH:MM",
                "end": "HH:MM",
                "activity": "short past-tense description of the new activity",
            }
        ],
    }
    return (
        "PLAN CHANGES FROM DIALOGUE. The external JSON response must contain life_update. "
        "Normally it is null. Change the plan only if the current incoming message specifically "
        "proposes moving, replacing, adding, or cancelling a future activity AND you agree to "
        "it in the reply. An ordinary question about plans, a hypothesis, a refusal, or a "
        "clarifying question is not an agreement: life_update=null. If the time is insufficient "
        "or unclear, ask for clarification without changing anything. Do not decide for the "
        "partner or declare a shared activity completed. Do not claim that the plan changed "
        "without life_update. Only today's future schedule may change; completed and current "
        "activities are immutable. For new or moved activities, choose a time strictly later "
        "than the current time, leaving room for response generation. Do not remove or change "
        "other activities; overlaps are prohibited. changes contains 1–8 operations: replace "
        "(existing id and new start/end/activity), cancel (only op and id), or add "
        "(op/start/end/activity, without id). activity describes only the persona's own "
        "simulated activity, never a biographical fact or the partner's action. Take source_id "
        "from the current incoming messages below, and base_revision and date from the current plan. "
        "Structure example (replace the content with the actual agreement):\n"
        + json.dumps(example, ensure_ascii=False)
        + "\nCurrent incoming messages for decision traceability:\n"
        + json.dumps(incoming, ensure_ascii=False)
    )


LIFE_CONTEXT_INSTRUCTIONS = (
    "PERMITTED DAY SIMULATION. These are fictional activities for the persona, separate "
    "from real biography. They may be discussed naturally in the first person without "
    "repeatedly disclaiming the simulation. Describe completed activities as finished, "
    "current as happening now, and planned only as an intention. The active schedule is "
    "only activities. Values in before within recent_plan_changes describe cancelled or "
    "replaced plans, not active activities. Do not say the whole day was empty if completed "
    "activities exist. Never present the future as the past. New future activities are "
    "allowed only as an explicit life_update agreed in the current dialogue. If no plan "
    "exists, current activities are unknown.\n"
)
