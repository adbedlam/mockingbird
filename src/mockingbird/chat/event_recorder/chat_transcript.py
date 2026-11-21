from __future__ import annotations

from pathlib import Path

from mockingbird.chat.presenter import reason_label


class ChatTranscript:
    def __init__(self, path: Path):
        self.path = path

    def append(self, at: str, stage: str, message: str, data: dict) -> None:
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(f"\n## {at} — {stage}\n\n{message}\n")
            if "text" in data:
                stream.write(
                    "\n"
                    + "\n".join("> " + line for line in str(data["text"]).splitlines())
                    + "\n"
                )
            if "reason" in data:
                stream.write(
                    "\n" + reason_label(data) + ": " + str(data["reason"]) + "\n"
                )
            if "request_path" in data:
                stream.write(
                    "\nContext and raw response: " + str(data["request_path"]) + "\n"
                )
