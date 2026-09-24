from mockingbird.models.ollama.model import OllamaModel

ollama = OllamaModel()

content = [{"role": "user", "content": "What is 2 + 2?"}]

print(ollama.generate(content))
