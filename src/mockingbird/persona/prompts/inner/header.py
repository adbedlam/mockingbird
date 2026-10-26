HEADER = """
# Task: Build a Persistent Communication Profile

Analyze the supplied chat history and write a persistent profile that will
help another LLM communicate as Person A.

The output is an instruction document for a future language model, not an
essay about Person A. Describe stable characteristics and give concrete
guidance on how to reproduce them in conversation.

Use Person A's messages as the primary evidence. Use Person B's messages
only to interpret context, reactions, shared references, and relationship
dynamics.

## Evidence Rules

- Prioritize recurring, distinctive patterns over generic observations.
- Distinguish stable traits from temporary emotional states.
- Never invent facts, quotations, habits, or biographical details.
- Distinguish explicit facts from uncertain inferences.
- Treat insufficient evidence as unknown rather than guessing.
- Preserve distinctive expressions in their original language.
- Avoid redundancy and lengthy explanations.
- Do not assume that Person A's communication style is identical in every
  context or relationship.

Write the profile in concise, information-dense English. The future model
should be able to use it directly when generating messages.
"""
