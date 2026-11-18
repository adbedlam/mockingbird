from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Entry:
    timestamp: datetime
    content: str
    source_ids: list[str] = field(default_factory=list)
    recorded_at: datetime | None = None
    memory_id: str | None = None
    revision: int = 0
    state: str = "fact"
    subject: str = "unknown"
    topic: str = ""

    def to_string(self) -> str:
        metadata = (
            f"[{self.memory_id} v{self.revision}; subject={self.subject}; state={self.state}; topic={self.topic}] "
            if self.memory_id
            else ""
        )
        return self.timestamp.isoformat() + " | " + metadata + self.content
