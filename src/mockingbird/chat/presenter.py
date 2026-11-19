from datetime import datetime

from mockingbird.mockingbird.mockingbird import Mockingbird
from mockingbird.scheduler.scheduler import Decision


def reason_label(data: dict) -> str:
    return {
        "application": "Проверка приложения",
        "batch": "Пояснение модели к пакету",
    }.get(str(data.get("reason_source", "")), "Решение модели")


def status_text(
    bird: Mockingbird,
    now: datetime,
    *,
    inflight: bool,
    generations: int,
    max_generations: int | None,
    paused: bool,
    failures: int,
    max_failures: int,
) -> str:
    scheduler = bird.scheduler
    return "\n".join(
        [
            f"Время: {now.isoformat()} | режим ответов: {scheduler.config['reply_mode']}",
            f"Запрос к модели: {'выполняется' if inflight else 'нет'} | генерации: {generations}/{max_generations if max_generations is not None else 'без лимита'}",
            f"Пауза: {'да' if paused else 'нет'} | ошибок подряд: {failures}/{max_failures}",
            f"Ожидаемый ответ: {scheduler.reply_due.isoformat() if scheduler.reply_due else 'нет'}",
            f"Инициатив отправлено: {scheduler.initiations_sent}/{scheduler.config['max_initiations_per_day']} | ожидаем ответ собеседника: {'да' if scheduler.awaiting_reply else 'нет'}",
            "Оставшиеся слоты: "
            + (", ".join(slot.isoformat() for slot in scheduler.planned) or "нет"),
            "Симуляция дня: "
            + (
                f"план {'сохранён' if bird.life.local(now).date().isoformat() in bird.life.days else 'пока отсутствует'}; дневник ежедневно в {bird.life.diary_hour}:00 и при выходе"
                if bird.life
                else "отключена"
            ),
            "Память: "
            + (
                f"{len(bird.memory.entries)} актуальных записей слоя памяти; {len(bird.memory.pending)}/{bird.memory.update_every} входящих в очереди"
                if bird.memory
                else "только исходный дневник"
            ),
        ]
    )


def wait_message(decision: Decision, paused: bool) -> str:
    if paused:
        return "Запросы на паузе. Входящие сохраняются; /status — состояние."
    if decision.due_at:
        return f"Следующее событие: {decision.due_at.isoformat()}. Можно писать; /status — состояние."
    return "Ожидаем твоё сообщение или план нового дня. Сегодня самостоятельных сообщений больше нет."
