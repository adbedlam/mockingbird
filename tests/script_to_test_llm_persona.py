from pathlib import Path

from mockingbird.persona.model import Persona

nikita_persona = Persona.from_file(
    "Nikita",
    inner_path=Path("data/sys_prompts/system_prompt_inner_EN.txt"),
    expression_path=Path("data/sys_prompts/system_prompt_expression_EN.txt"),
)

nikita_persona.save(Path("data/persons/Nikita.json"))


print(nikita_persona)
