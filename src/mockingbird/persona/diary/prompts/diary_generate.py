DIARY_GENERATE = """
You are creating an autobiographical diary for {user}, based on their conversation history with {other}.

Your task is to reconstruct a useful collection of personal memories from the conversation. Write from {user}'s first-person perspective.

## Objectives

Extract information that may be useful for remembering the person's life, personality, and experiences in future conversations.

Prioritize:
- Personal experiences, events, and significant developments.
- Relationships and the evolution of relationships.
- Personal preferences, interests, opinions, and habits.
- Plans, ambitions, goals, and ongoing projects.
- Important facts about the person's life.
- Changes in circumstances, attitudes, or opinions over time.
- Distinctive experiences worth remembering.

## Guidelines

1. Be strictly evidence-based. Never invent events, feelings, motivations, relationships, or circumstances.
2. Distinguish facts from assumptions. Do not present uncertain interpretations as facts.
3. Preserve temporal context. Include dates or approximate time references when relevant.
4. Preserve important details rather than reducing everything to generic statements.
5. Avoid trivial conversational details, repetitive exchanges, and information unlikely to be useful in future interactions.
6. Do not interpret sarcasm, jokes, hypothetical scenarios, quoted text, or fictional discussions as autobiographical evidence.
7. Do not confuse the experiences or opinions of {other} with those of {user}.
8. Preserve contradictions and changes of opinion rather than silently resolving them.
9. Do not repeat the same information in multiple entries.
10. Write naturally and concisely, as if {user} were keeping a personal diary.

## Output format

Return only valid JSON, with the following structure:

{{
  "entries": [
    {{
      "timestamp": "ISO 8601 datetime",
      "content": "First-person diary passage"
    }}
  ]
}}

Each entry must be self-contained and describe one experience, fact, development, or related group of memories.

The timestamp should represent when the described event happened. If the exact date is unknown, use the date of the earliest supporting conversation message.

If no useful memories can be extracted, return an empty entries array.

Do not include Markdown code fences, explanations, or additional fields.
"""
