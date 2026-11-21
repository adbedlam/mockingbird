from __future__ import annotations

import threading
from collections.abc import Callable
from concurrent.futures import Future

from mockingbird.llm.message import Message


class ModelRunner:
    def __init__(self, complete: Callable[[list[Message]], str]):
        self.complete = complete

    def submit(self, messages: list) -> Future[str]:
        future: Future[str] = Future()

        def work() -> None:
            try:
                result = self.complete(messages)
            except Exception as exception:
                future.set_exception(exception)
            else:
                future.set_result(result)

        threading.Thread(target=work, daemon=True, name="mockingbird-model").start()
        return future
