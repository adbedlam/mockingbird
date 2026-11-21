from __future__ import annotations

import logging
from typing import Any

import httpx

from mockingbird.interfaces.telegram.errors import (
    BotAPIError,
    BotNetworkError,
    SetupError,
)

_ALLOWED_METHODS = frozenset(
    {
        "getMe",
        "getWebhookInfo",
        "getUpdates",
        "getBusinessConnection",
        "sendMessage",
    }
)


class TelegramBotAPI:
    def __init__(self, token: str, transport: httpx.AsyncBaseTransport | None = None):
        if not token or ":" not in token:
            raise SetupError("Заполни TELEGRAM_BOT_TOKEN в локальном .env.")
        self._token = token
        # httpx INFO messages include the token-bearing Bot API URL.
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(25, connect=10),
            follow_redirects=False,
            transport=transport,
        )

    async def call(self, method: str, **params: Any) -> Any:
        self._validate_method(method)

        response = await self._post(method, params)
        data = self._parse_response(response, method)

        self._raise_api_error_if_needed(data, response, method)

        return data.get("result")

    def _validate_method(self, method: str) -> None:
        if method not in _ALLOWED_METHODS:
            raise ValueError("Unsupported Bot API method")

    async def _post(
        self,
        method: str,
        params: dict[str, Any],
    ) -> httpx.Response:
        try:
            return await self.client.post(
                f"https://api.telegram.org/bot{self._token}/{method}",
                json=params,
            )
        except httpx.TransportError as exception:
            raise BotNetworkError(
                type(exception).__name__,
                method=method,
            ) from None

    def _parse_response(
        self,
        response: httpx.Response,
        method: str,
    ) -> dict[str, Any]:
        try:
            data = response.json()
        except ValueError:
            raise BotNetworkError(
                f"InvalidJSON (HTTP {response.status_code})",
                method=method,
            ) from None

        if not isinstance(data, dict):
            raise BotNetworkError(
                "Invalid Bot API response",
                method=method,
            )

        return data

    def _raise_api_error_if_needed(
        self,
        data: dict[str, Any],
        response: httpx.Response,
        method: str,
    ) -> None:
        if data.get("ok") is True:
            return

        description = str(data.get("description", "Bot API rejected request")).replace(
            self._token, "[REDACTED]"
        )

        parameters = data.get("parameters") or {}

        raise BotAPIError(
            int(data.get("error_code", response.status_code)),
            description,
            int(parameters.get("retry_after", 0)),
        )

    async def close(self) -> None:
        await self.client.aclose()
