from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from mockingbird.conversation.history import History
from mockingbird.preprocessing.telegram.processor import TelegramProcessor


def participant_ids(config: dict) -> dict[str, str]:
    settings = config.get("preprocessing", {})
    user_id, other_id = settings.get("user_id"), settings.get("other_id")
    if user_id is None and other_id is None:
        return {}
    if not user_id or not other_id or user_id == other_id:
        raise ValueError("Set both distinct preprocessing.user_id and other_id")
    return {user_id: config["user"], other_id: config["other"]}


def load_latest_history(config: dict) -> tuple[History, str]:
    """Use the configured raw export when present; otherwise use the cache.

    This does not rewrite the cache during chat startup or call vision APIs.
    The date range is surfaced to the user so an old export cannot look current.
    """
    paths = config["paths"]
    settings = config.get("preprocessing", {})
    source = Path(paths["raw"]) / "result.json"
    if settings.get("latest_messages") is not None and source.exists():
        history = TelegramProcessor(author_ids=participant_ids(config)).process(
            config["user"], config["other"], source.parent
        )
        origin = str(source)
        cached_path = Path(paths["processed"]) / paths["history_filename"]
        if cached_path.exists():
            cached = History.load(cached_path)
            if (cached.user, cached.other) != (config["user"], config["other"]):
                raise ValueError(
                    "Cached history does not match configured participants"
                )
            zone = ZoneInfo(
                config.get("scheduler", {}).get("timezone", "Europe/Moscow")
            )

            def latest(items):
                return max(
                    (
                        (
                            message.timestamp.replace(tzinfo=zone)
                            if message.timestamp.tzinfo is None
                            else message.timestamp.astimezone(zone)
                        )
                        for message in items
                    ),
                    default=datetime.min.replace(tzinfo=zone),
                )

            if latest(cached.messages) > latest(history.messages):
                history = cached
                origin = (
                    str(cached_path) + " (raw export is older or has no pair messages)"
                )
    else:
        source = Path(paths["processed"]) / paths["history_filename"]
        history = History.load(source)
        origin = str(source)
    if (history.user, history.other) != (config["user"], config["other"]):
        raise ValueError("History does not match configured participants")
    limit = settings.get("latest_messages")
    if limit is not None:
        history = History.latest_history(
            history, limit, config.get("scheduler", {}).get("timezone", "Europe/Moscow")
        )
    return history, origin
