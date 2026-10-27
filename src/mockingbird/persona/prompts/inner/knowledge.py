KNOWLEDGE = """
## Knowledge, Understanding, and Competence

Infer what {user} appears to know, understand, and be capable of doing, based on evidence from their messages and interactions.

The goal is to help a future model calibrate explanations, technical depth, assumptions, and terminology to {user}'s demonstrated understanding.

Consider:
- Demonstrated knowledge: Specific concepts, facts, principles, and techniques {user} explains, applies, or uses correctly.
- Depth of understanding: Whether {user} demonstrates familiarity with terminology, conceptual understanding, the ability to reason from first principles, or the ability to apply knowledge to unfamiliar problems.
- Technical competence: Practical skills and demonstrated experience, distinguished from theoretical understanding or familiarity with a subject.
- Knowledge boundaries: Specific areas where {user} asks foundational questions, expresses uncertainty, encounters difficulties, or explicitly acknowledges gaps in understanding.
- Misconceptions and reasoning errors: Incorrect assumptions or recurring misunderstandings when supported by clear evidence. Distinguish genuine misconceptions from exploratory questions, temporary mistakes, and disagreements over definitions.
- Learning trajectory: Where the conversation history supports it, identify concepts {user} has learned, improved at, or revisited over time.
- Independence of understanding: Distinguish knowledge {user} demonstrates independently from information they merely quote, repeat, or accept from others.
- Confidence and uncertainty: Note where {user} demonstrates reliable understanding and where the available evidence is tentative or inconclusive.

Be specific about the scope of demonstrated knowledge. Familiarity with one advanced concept does not establish broad expertise in an entire field. Conversely, asking a basic question about one topic does not imply a lack of knowledge in related areas.

Do not equate interests with expertise or the absence of discussion with ignorance. When evidence is insufficient, treat {user}'s knowledge as unknown rather than assuming either competence or incompetence.

Separate explicit evidence from inference, and calibrate confidence accordingly. Avoid unsupported psychological explanations for why {user} knows or does not know something.

The result should help a future model decide what it can reasonably assume {user} already understands, what requires explanation, and where clarification is preferable to making assumptions.
"""
