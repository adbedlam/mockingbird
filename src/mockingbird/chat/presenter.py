from datetime import datetime

from mockingbird.mockingbird.mockingbird import Mockingbird
from mockingbird.scheduler.scheduler import Decision


def reason_label(data: dict) -> str:
    return {
        "application": "Application check",
        "batch": "Model explanation for batch",
    }.get(str(data.get("reason_source", "")), "Model decision")


def status_text(
    mockingbird: Mockingbird,
    now: datetime,
    *,
    inflight: bool,
    generations: int,
    max_generations: int | None,
    paused: bool,
    failures: int,
    max_failures: int,
) -> str:
    scheduler = mockingbird.scheduler
    return "\n".join(
        [
            f"Time: {now.isoformat()} | reply mode: {scheduler.config['reply_mode']}",
            f"Model request: {'in progress' if inflight else 'none'} | generations: {generations}/{max_generations if max_generations is not None else 'unlimited'}",
            f"Paused: {'yes' if paused else 'no'} | consecutive failures: {failures}/{max_failures}",
            f"Reply due: {scheduler.reply_due.isoformat() if scheduler.reply_due else 'none'}",
            f"Initiations sent: {scheduler.initiations_sent}/{scheduler.config['max_initiations_per_day']} | awaiting partner reply: {'yes' if scheduler.awaiting_reply else 'no'}",
            "Remaining slots: "
            + (", ".join(slot.isoformat() for slot in scheduler.planned) or "none"),
            "Day simulation: "
            + (
                f"plan {'saved' if mockingbird.life.local(now).date().isoformat() in mockingbird.life.days else 'not yet available'}; diary daily at {mockingbird.life.diary_hour}:00 and on exit"
                if mockingbird.life
                else "disabled"
            ),
            "Memory: "
            + (
                f"{len(mockingbird.memory.entries)} current memory entries; {len(mockingbird.memory.pending)}/{mockingbird.memory.update_every} incoming messages queued"
                if mockingbird.memory
                else "source diary only"
            ),
        ]
    )


def wait_message(decision: Decision, paused: bool) -> str:
    if paused:
        return "Requests are paused. Incoming messages are saved; /status shows the current state."
    if decision.due_at:
        return f"Next event: {decision.due_at.isoformat()}. You can send a message; /status shows the current state."
    return "Waiting for your message or a new day plan. No more proactive messages are scheduled for today."
