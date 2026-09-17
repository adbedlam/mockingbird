from typing import Protocol


class Model(Protocol):
    def generate(self, messages: list[dict[str, str]], *args, **kwargs): ...
