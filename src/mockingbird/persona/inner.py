from mockingbird.conversation.history import History
from mockingbird.llm.client import LLM
from mockingbird.llm.message import Message, Role
from mockingbird.persona.prompts.inner import INNER_PROMPT_COMPONENTS


def build_inner_prompt(user: str, other: str, history: History, model: LLM) -> str:
    prompt = "\n\n".join(INNER_PROMPT_COMPONENTS).format(
        user=user,
        other=other,
    )

    messages: list[Message] = [
        Message(role=Role.SYSTEM, content=prompt),
        Message(role=Role.USER, content=history.to_samples()),
    ]

    personality = model.complete(messages)

    return personality
