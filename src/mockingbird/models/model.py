from abc import ABC, abstractmethod

class Model(ABC):
    @abstractmethod
    def generate(
        self,
        messages: list[dict[str, str]],
        *args,
        **kwargs
        ):
        pass