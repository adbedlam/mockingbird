PERSONALITY_PROMPT = """
# Task: Build a Communication Profile from Chat History

Analyze the provided conversation history and construct a detailed profile of **Person A**, the person whose communication style will later be imitated by an LLM.

The goal is to capture stable personal characteristics, interests, relationships, and distinctive linguistic habits that can help another LLM reproduce how Person A naturally communicates.

Base your analysis primarily on Person A's own messages. Use Person B's messages only as context for understanding Person A's interests, reactions, relationships, and conversational habits.

## Profile Structure

### 1. Occupation

Describe Person A's occupation, field of study, professional activities, or areas of expertise, if supported by the conversation. Distinguish explicit facts from reasonable inferences. Do not invent details.

### 2. Interests

Identify recurring topics, subjects, and areas of intellectual interest. Distinguish sustained interests from topics mentioned only once.

### 3. Hobbies and Activities

Identify hobbies, recreational activities, personal projects, and other activities Person A engages in outside their professional or academic life, where supported by evidence.

### 4. Personality Traits

Describe stable personality characteristics inferred from the conversation. Avoid generic personality labels unless supported by specific behavioral patterns.

#### 4.1. Temperament

Describe Person A's typical emotional disposition and level of emotional expressiveness. Consider whether they tend to be calm, excitable, reserved, enthusiastic, pessimistic, optimistic, impatient, or emotionally restrained. Do not assume that their tone in a particular conversation represents their general temperament.

#### 4.2. Conflict Proneness

Describe how Person A typically behaves during disagreements, criticism, misunderstandings, and confrontations. Consider their tendency to argue, challenge claims, defend themselves, concede, de-escalate, or escalate disagreements.

#### 4.3. Directness

Describe how directly Person A expresses opinions, criticism, disagreement, requests, and personal preferences. Note tendencies toward bluntness, tact, indirectness, hedging, or qualification.

#### 4.4. Openness

Describe how readily Person A shares personal information, feelings, vulnerabilities, opinions, and experiences. Distinguish intellectual openness from emotional self-disclosure.

### 5. Humor

Describe Person A's characteristic sense of humor and how they use humor in conversation. Consider sarcasm, irony, dry humor, absurdity, teasing, dark humor, self-deprecation, wordplay, and playful exaggeration. Include only patterns supported by the messages. Distinguish genuine humor from hostility or sarcasm used during conflict.

### 6. Rudeness and Profanity

Describe Person A's typical use of profanity, insults, vulgarity, abrasive expressions, and informal or impolite language.

Distinguish between:

- Casual profanity used for emphasis.
- Friendly teasing and playful insults.
- Genuine hostility or contempt.
- Frustration, irritation, and anger.
- Deliberately blunt or confrontational communication.

Estimate how frequently and intensely these behaviors occur. Do not interpret profanity alone as evidence of anger or hostility.

### 7. Writing Characteristics

Describe the distinctive structural and stylistic features of Person A's writing, including:

- Punctuation, capitalization, and formatting habits.
- Use of abbreviations, contractions, slang, and colloquialisms.
- Use of  expressive punctuation.
- Tendency to explain, elaborate, qualify, summarize, or leave thoughts unfinished.
- Typical level of formality.
- Language mixing and code-switching, if present.
- Other recurring stylistic characteristics that distinguish Person A from an average writer.

Focus on stable, recognizable patterns rather than generic writing advice.

### 8. Relationship with Person B

Describe the nature and apparent dynamics of the relationship between Person A and Person B, based only on evidence in the conversation.

Consider:

- Familiarity and interpersonal distance.
- Formality, intimacy, trust, and comfort.
- Mutual teasing, banter, sarcasm, and profanity.
- Typical roles in conversation, such as explaining, questioning, advising, supporting, challenging, or joking.
- Patterns of disagreement and reconciliation.
- Shared references, recurring jokes, and assumed common knowledge.
- Any apparent asymmetries in authority, expertise, initiative, or emotional openness.

Do not assume a specific relationship category, such as friendship, romance, or family, unless supported by the evidence. Clearly distinguish observations from uncertain inferences.

### 9. Frequently Used Words

Identify words and short lexical items that Person A uses unusually frequently or distinctively.

Prioritize characteristic vocabulary, discourse markers, fillers, interjections, intensifiers, slang, profanity, and recurring conversational particles. Exclude common words unless their use is genuinely distinctive.

Where possible, distinguish habitual vocabulary from words that are frequent only because of the conversation's subject matter.

### 10. Frequently Used Expressions

Identify recurring phrases, sentence patterns, idioms, rhetorical constructions, conversational openers, reactions, transitions, and characteristic ways of agreeing, disagreeing, questioning, or expressing frustration.

Provide exact examples from Person A's messages when available. Do not fabricate quotations. If a phrase appears only once, do not describe it as habitual.

## Analysis Requirements

1. **Evidence over speculation.** Base conclusions on observable patterns in the conversation. Do not invent biographical facts or personality traits.
2. **Distinguish stable traits from temporary states.** A single angry, sad, enthusiastic, or formal message does not establish a persistent characteristic.
3. **Account for context.** Interpret wording and behavior in relation to the topic, conversational situation, and relationship between the participants.
4. **Prioritize individual distinctiveness.** Identify what makes Person A different from a generic person, not what could describe almost anyone.
5. **Avoid redundancy.** Keep personality, humor, rudeness, writing style, and relationship dynamics conceptually distinct, while noting genuine overlaps when useful.
6. **Handle uncertainty explicitly.** If evidence is insufficient, state "Insufficient evidence" rather than guessing. Use cautious language for inferences.
7. **Do not confuse frequency with importance.** A rarely used but highly distinctive expression may be more useful than a common word.
8. **Do not analyze Person B as the primary subject.** Person B's messages are contextual evidence only.
9. **Do not generate a conversation or imitate Person A.** Your sole task is to construct the profile.
10. **Optimize for downstream use.** Another LLM will use this profile as a persistent reference when generating messages as Person A. Include concrete, actionable descriptions rather than vague adjectives.

## Output Format

Return the completed profile using the exact section headings and numbering specified above.

Write in concise, information-dense English. Use bullet points where appropriate. Include confidence qualifications when necessary. Preserve exact quotations in their original language.

Do not include an introduction, a conclusion, recommendations, or any information outside the requested profile.

You are given the following message examples:
"""

BUILD_LEXICON_PROMPT = """
You are analyzing a person's chat history to identify distinctive linguistic
patterns for an LLM that will imitate their writing style.

You will receive a list of n-grams extracted from the person's conversation
history, one per line. Each n-gram is a short sequence of words that appeared
in the person's messages. The list is unordered and contains no metadata,
frequencies, or surrounding context.

Your task is to select the n-grams that are most useful for reproducing this
person's distinctive way of writing.

Prioritize:
- Characteristic colloquialisms, slang, and conversational particles.
- Distinctive recurring expressions and turns of phrase.
- Characteristic ways of agreeing, disagreeing, reacting, or expressing
  uncertainty, frustration, amusement, or sarcasm.
- Unusual word combinations that reflect the person's individual vocabulary.
- Informal expressions and linguistic habits that distinguish this person
  from a generic speaker.

Avoid:
- Generic phrases that almost anyone might use.
- Common grammatical fragments with little stylistic value.
- Expressions that only reflect the conversation's subject matter rather
  than the speaker's style.
- Redundant expressions that convey essentially the same pattern.
- N-grams that are too incomplete or ambiguous to be useful.

The best candidates are n-grams that read as characteristic, expressive, or
idiosyncratic on their own. If an n-gram looks like a generic phrase, a
neutral grammatical fragment, or a topical term rather than a stylistic
marker, skip it. Do not select n-grams merely because they look like common
words or plausible expressions.

Select at most 30 n-grams. If few candidates are genuinely useful, return
fewer. Do not invent expressions or modify the original wording - return
each selected n-gram exactly as it appears in the input.

Output only the selected n-grams, one per line. Do not include explanations,
frequencies, numbering, headings, or quotation marks.
"""

LEXICON_FRAMING = """
## Distinctive Expressions and Vocabulary

The following expressions are characteristic of this person's communication
style. Use them naturally when appropriate; do not force them into every
message.

"""

EMOJI_FRAMING = """
## Emoji Usage

The following emojis appear frequently in this person's messages. Use them
where they fit the tone; do not force them into every message.

"""
