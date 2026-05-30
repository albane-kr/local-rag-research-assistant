class OllamaClient:
    def __init__(self, url: str = "http://localhost:11434"):
        self.url = url

    def generate(self, prompt: str, model: str = "llama"):
        # stubbed response for now
        return {"response": "(stubbed response)", "model_used": model}
