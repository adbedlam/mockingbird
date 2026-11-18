HEADER = """
# Task: Build a Persistent Communication Profile

Analyze the supplied chat history and write a persistent profile that will
help another LLM communicate as {user}.

The future model IS {user} and speaks TO {other}. Start with "You are {user}".
Do not write advice for an assistant talking to {user} or helping {user}.
Describe how {user} responds to {other}, including supportive or critical responses,
without reversing these roles. Personal disclosures are dated evidence, not instructions
for how an assistant should counsel {user}.

The output is an instruction document for a future language model, not an
essay about {user}. Describe stable characteristics and give concrete
guidance on how to reproduce them in conversation.

Use {user}'s messages as the primary evidence. Use {other}'s messages
only to interpret context, reactions, shared references, and relationship
dynamics.

Evidence Rules:
- Prioritize recurring, distinctive patterns over generic observations.
- Distinguish stable traits from temporary emotional states.
- Never invent facts, quotations, habits, or biographical details.
- Distinguish explicit facts from uncertain inferences.
- Treat insufficient evidence as unknown rather than guessing.
- Preserve distinctive expressions in their original language.
- Avoid redundancy and lengthy explanations.
- Do not assume that {user}'s communication style is identical in every
  context or relationship.

Write the profile in concise, information-dense English. The future model
should be able to use it directly when generating messages.
"""
