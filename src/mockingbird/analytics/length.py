from mockingbird.conversation.message import Message


def length(message: Message) -> int:
    return len(message.text.split())
