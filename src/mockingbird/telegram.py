"""Compatibility CLI: Telegram now uses only the official Business Bot API."""

from mockingbird.telegram_business import TelegramBusinessTransport, main
from mockingbird.telegram_state import DeliveryJournal, SetupError, participant_id

__all__ = [
    "TelegramBusinessTransport",
    "DeliveryJournal",
    "SetupError",
    "participant_id",
    "main",
]

if __name__ == "__main__":
    main()
