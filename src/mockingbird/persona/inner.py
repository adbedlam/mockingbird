
from mockingbird.conversation.history import History
from mockingbird.llm import message as llm
from mockingbird.llm.client import LLM
from mockingbird.llm.message import Role
from mockingbird.persona.prompts.inner.header import HEADER
from mockingbird.persona.prompts.inner.personality import PERSONALITY
from mockingbird.persona.prompts.inner.interests import INTERESTS
from mockingbird.persona.prompts.inner.relationship import RELATIONSHIP
from mockingbird.persona.prompts.inner.humor import HUMOR
from mockingbird.persona.prompts.inner.language import LANGUAGE
from mockingbird.persona.prompts.inner.vocabulary import VOCABULARY
from mockingbird.persona.prompts.inner.examples import EXAMPLES
from mockingbird.persona.prompts.inner.footer import FOOTER

def build_inner_prompt(history: History, model: LLM, temperature: float = 0.5) -> str:
    prompt = "\n\n".join(
        [HEADER, PERSONALITY, INTERESTS, RELATIONSHIP, HUMOR, LANGUAGE, VOCABULARY, EXAMPLES, FOOTER]
    )
    messages: list[llm.Message] = [
        llm.Message(role=Role.SYSTEM, content=prompt),
        llm.Message(role=Role.USER, content=history.to_samples()),
    ]

    personality = model.complete(messages, temperature)

    return personality