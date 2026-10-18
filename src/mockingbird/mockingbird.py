# from mockingbird.llm.model import Model
# from mockingbird.persona.persona import Persona


# class Mockingbird:
#     def __init__(self, persona: Persona, model: Model):
#         self.persona = persona
#         self.model = model
#         self.history = [{"role": "system", "content": persona.generate_system_prompt()}]

#     def mock(self) -> None:
#         while True:
#             try:
#                 user_input = input("user> ").strip()
#             except (EOFError, KeyboardInterrupt):
#                 break

#             if not user_input:
#                 continue

#             self.history.append({"role": "user", "content": user_input})

#             try:
#                 response = self.model.complete(self.history)
#             except Exception as exception:
#                 print(f"[error] {exception}")

#                 self.history.pop()

#                 continue

#             self.history.append({"role": "assistant", "content": response.text})

#             print(f"mockingbird> {response.text}")
