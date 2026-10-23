import pandas as pd

from mockingbird.analytics.emoji import emoji_count
from mockingbird.analytics.length import length
from mockingbird.conversation.history import History


def analyze(
    history: History,
    user: str,
) -> pd.DataFrame:
    messages = history.messages

    return pd.DataFrame(
        {
            "length": [length(message) for message in messages],
            "num_emojis": [emoji_count(message) for message in messages],
            "from_user": [message.author == user for message in messages],
        }
    )
