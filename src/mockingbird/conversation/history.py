from dataclasses import dataclass

from mockingbird.conversation.message import Message
from mockingbird.conversation.username import Username


@dataclass
class History:
    """
    Plain list of messages.
    """

    user: Username
    other: Username
    messages: list[Message]

    def users_messages(self) -> list[Message]:
        return [message for message in self.messages if message.author == self.user]
