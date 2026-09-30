from pathlib import Path


class CorePersonality:
    inner_prompt: str

    def __init__(self, inner_path: Path):
        self.inner_prompt = inner_path.read_text(encoding="utf-8")
