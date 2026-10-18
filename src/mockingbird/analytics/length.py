from mockingbird.conversation.message import Message


def average_length(messages: list[Message]) -> float:
    return sum(len(message.text) for message in messages) / len(messages)
