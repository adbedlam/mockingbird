from __future__ import annotations

import threading
from collections.abc import Callable
from concurrent.futures import Future

from mockingbird.chat.tasks import ModelResult


class ModelRunner:
    def __init__(self, complete: Callable[[list], str]):
        self.complete = complete

    def submit(self, messages: list) -> Future[ModelResult]:
        future: Future[ModelResult] = Future()

        def work() -> None:
            try:
                result = ModelResult(raw=self.complete(messages))
            except Exception as exc:
                status = getattr(exc, "status_code", None)
                result = ModelResult(
                    error=type(exc).__name__ + (f" (HTTP {status})" if status else "")
                )
            future.set_result(result)

        threading.Thread(target=work, daemon=True, name="mockingbird-model").start()
        return future
