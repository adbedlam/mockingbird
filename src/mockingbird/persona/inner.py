from mockingbird.config import load_config
from mockingbird.conversation.history import History
from mockingbird.llm import message as llm
from mockingbird.llm.client import LLM
from mockingbird.llm.message import Role
from mockingbird.persona.prompts.inner import INNER_PROMPT_COMPONENTS


def build_inner_prompt(history: History, model: LLM, temperature: float = 0.5) -> str:
    config = load_config()

    prompt = "\n\n".join(INNER_PROMPT_COMPONENTS).format(
        user=config["user"],
        other=config["other"],
    )

    messages: list[llm.Message] = [
        llm.Message(role=Role.SYSTEM, content=prompt),
        llm.Message(role=Role.USER, content=history.to_samples()),
    ]

    personality = model.complete(messages, temperature)

    return personality
