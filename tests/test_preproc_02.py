from pathlib import Path

from mockingbird.preprocessing.telegram.by_user_processor import UserTelegramProcessor

processor = UserTelegramProcessor()


res = processor.process(
    data_path=Path("../mockingbird/data/result.json"),
)

print(len(res))

print(res[:10])
