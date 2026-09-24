import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Self


@dataclass
class Persona:
    name: str
    inner_prompt: str
    expression_prompt: str

    @classmethod
    def from_file(cls, name, inner_path: Path, expression_path: Path) -> Self:
        inner_prompt = inner_path.read_text(encoding="utf-8")
        expression_prompt = expression_path.read_text(encoding="utf-8")

        return cls(
            name=name,
            inner_prompt=inner_prompt,
            expression_prompt=expression_prompt,
        )

    def save(self, path: Path):
        path.write_text(
            json.dumps(
                asdict(self),
                indent=4,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> Self:
        data = json.loads(path.read_text(encoding="utf-8"))

        return cls(**data)
