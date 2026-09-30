from typing import Protocol

from mockingbird.datacls import ConversationHistory, Response


class Model(Protocol):
    def __init__(self, model: str, *args, **kwargs): ...

    def generate(self, history: ConversationHistory, *args, **kwargs) -> Response: ...
