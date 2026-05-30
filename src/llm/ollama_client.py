import requests
from typing import Optional


class OllamaClient:
    """Client for calling local Ollama LLM server."""

    def __init__(self, url: str = "http://localhost:11434"):
        self.url = url.rstrip("/")

    def generate(
        self,
        prompt: str,
        model: str = "llama2",
        stream: bool = False,
    ) -> dict:
        """
        Call Ollama generate endpoint.
        Returns {"response": str, "model": str, "done": bool} on success.
        Raises RuntimeError if Ollama is unreachable.
        """
        endpoint = f"{self.url}/api/generate"
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": stream,
        }

        try:
            response = requests.post(endpoint, json=payload, timeout=300)
            response.raise_for_status()
            data = response.json()
            return {
                "response": data.get("response", ""),
                "model": data.get("model", model),
                "done": data.get("done", True),
            }
        except requests.exceptions.ConnectionError:
            raise RuntimeError(
                f"Cannot connect to Ollama at {self.url}. "
                "Ensure Ollama is running: ollama serve"
            )
        except requests.exceptions.Timeout:
            raise RuntimeError(f"Ollama request timed out after 300s")
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Ollama request failed: {e}")
