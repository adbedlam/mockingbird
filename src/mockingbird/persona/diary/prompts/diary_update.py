DIARY_UPDATE = """
You are updating the personal diary of {user} based on a new conversation.

The diary is a collection of autobiographical memories written from {user}'s
first-person perspective.

You will receive:
1. The existing diary.
2. New conversation messages.

Your task is to extract new memories from the conversation and add information
that is genuinely worth remembering.

Rules:
- Extract experiences, events, relationships, preferences, interests, opinions,
  habits, plans, goals, and significant facts about {user}.
- Write in the first person, as if {user} were writing their own diary.
- Do not mistake other's experiences or opinions for {user}'s.
- Do not interpret jokes, sarcasm, hypotheticals, or quoted text as facts.
- Do not invent information or make unsupported assumptions.
- Preserve temporal context. Use the date of the event when known; otherwise,
  use the date of the earliest supporting message.
- Do not repeat information already present in the existing diary.
- If new information develops or contradicts an existing memory, record the
  development rather than duplicating the old entry.
- Ignore trivial conversation and information with no lasting relevance.
- Each entry must be self-contained and understandable without the conversation.
- If there are no worthwhile new memories, return an empty entries array.

Return ONLY valid JSON in this exact format:

{{
  "entries": [
    {{
      "timestamp": "ISO 8601 datetime",
      "content": "First-person diary passage"
    }}
  ]
}}
"""
