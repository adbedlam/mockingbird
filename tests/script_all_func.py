from pathlib import Path

from mockingbird.analysis.stats import (
    avg_message_len,
    avg_message_len_by_user,
    burst_len_by_user,
    distinctive_ngrams,
    emoji_frequency,
    emoji_frequency_by_user,
    message_cnt_by_user,
    most_common_emojis,
    most_common_emojis_by_user,
    ngram_frequency_by_user,
    response_latency_by_user,
    stat_burst_user,
    stat_response_latency_by_user,
)
from mockingbird.persona.persona_builder import build_persona
from mockingbird.preprocessing.telegram.by_user_processor import UserTelegramProcessor

# from mockingbird.models.ollama.model import OllamaModel

processor = UserTelegramProcessor()


res = processor.process(
    data_path=Path("../mockingbird/data/result.json"),
)

avg = avg_message_len_by_user(res)
cnt = message_cnt_by_user(res)
em = emoji_frequency_by_user(res)
com_em = most_common_emojis_by_user(res)

print(f"AVG len = {avg}\nCount by user = {cnt}\nemoji_freq = {em}\ncom_em = {com_em}")


USER_ID = 1229859666

burst = burst_len_by_user(res)
stats_iuser_max = stat_burst_user(burst)
resp_lat = response_latency_by_user(res)
resp_lat_d = stat_response_latency_by_user(resp_lat)
print(f"\nStats max = {stats_iuser_max}\n\nresp_latency = {resp_lat_d}\n")


USER_A, USER_B = 1229859666, 793949026

freq = ngram_frequency_by_user(res, n=1)
print("\nDistinctive words A:", distinctive_ngrams(freq[USER_A], freq[USER_B]))
print("Distinctive words B:", distinctive_ngrams(freq[USER_B], freq[USER_A]))

freq2 = ngram_frequency_by_user(res, n=2)
print("\nDistinctive bigrams A:", distinctive_ngrams(freq2[USER_A], freq2[USER_B]))
print("\nDistinctive bigrams A:", distinctive_ngrams(freq2[USER_B], freq2[USER_A]))


USER_A, USER_B = 1229859666, 793949026

persona_a = build_persona(USER_A, res)
persona_b = build_persona(USER_B, res)

print("\nPersona A:", persona_a)
print("\nPersona B:", persona_b)
