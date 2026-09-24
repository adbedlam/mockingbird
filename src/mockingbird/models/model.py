from typing import Protocol


class Model(Protocol):
    def __init__(self, model: str, base_url: str): ...

    def generate(self, messages: list[dict[str, str]], *args, **kwargs): ...
