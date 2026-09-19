from mockingbird.analysis.stats import (
    avg_message_len_by_user,
    burst_len_by_user,
    distinctive_ngrams,
    emoji_frequency_by_user,
    most_common_emojis_by_user,
    ngram_frequency_by_user,
    response_latency_by_user,
    stat_burst_user,
    stat_response_latency_by_user,
)
from mockingbird.datacls import Message
from mockingbird.persona.model import (
    BehavioralPatterns,
    CommunicationStyle,
    Persona,
)


def build_persona(user_id: int, messages: list[Message]):

    other_user_id = next(
        uid for uid in set(m.user_id for m in messages) if uid != user_id
    )

    avg_message_len = avg_message_len_by_user(messages=messages)[user_id]
    emoji_frequency = emoji_frequency_by_user(messages=messages)[user_id]
    most_common_emojis = most_common_emojis_by_user(messages=messages)[user_id]
    burst_stats = stat_burst_user(burst_len_by_user(messages))[user_id]
    response_latency_stats = stat_response_latency_by_user(
        response_latency_by_user(messages)
    )[user_id]

    freq = ngram_frequency_by_user(messages, n=1)
    distinctive_words = distinctive_ngrams(freq[user_id], freq[other_user_id])

    freq2 = ngram_frequency_by_user(messages, n=2)
    distinctive_phrases = distinctive_ngrams(freq2[user_id], freq2[other_user_id])

    style = CommunicationStyle(
        avg_message_len=avg_message_len,
        emoji_frequency=emoji_frequency,
        most_common_emojis=most_common_emojis,
        burst_stats=burst_stats,
        response_latency_stats=response_latency_stats,
        distinctive_words=distinctive_words,
        distinctive_phrases=distinctive_phrases,
        sample_messages=["hi"],
    )

    return Persona(
        user_id=user_id,
        message_count=sum(1 for m in messages if m.user_id == user_id),
        communication_style=style,
    )
