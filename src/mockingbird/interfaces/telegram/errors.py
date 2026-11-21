from mockingbird.observability import redact


class BotAPIError(RuntimeError):
    def __init__(self, code: int, description: str, retry_after: int = 0):
        self.code, self.retry_after = code, retry_after
        super().__init__(redact(description))


class BotNetworkError(RuntimeError):
    def __init__(self, detail: str, *, method: str = ""):
        self.detail, self.method = detail, method
        super().__init__(f"{method}: {detail}" if method else detail)


class SetupError(RuntimeError):
    """An error occured during setup."""
