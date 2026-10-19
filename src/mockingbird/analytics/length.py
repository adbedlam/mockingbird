from mockingbird.conversation.message import Message


def length(message: Message) -> int:
    return len(message.text.split())


def average_length(messages: list[Message]) -> float:
    if not messages:
        return 0.0
    return sum(len(message.text.split()) for message in messages) / len(messages)
