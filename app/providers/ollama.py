import json

import httpx

from app import config
from app.providers.errors import ProviderError

_client = httpx.Client(base_url=config.OLLAMA_URL, timeout=300)


def _post(path: str, payload: dict) -> dict:
    try:
        r = _client.post(path, json=payload)
        r.raise_for_status()
        return r.json()
    except httpx.ConnectError:
        raise ProviderError("Ollama est injoignable. Est-il lancé ? (ollama serve)")
    except httpx.HTTPStatusError as e:
        raise ProviderError(f"Erreur Ollama {e.response.status_code} : {e.response.text}")
    except httpx.HTTPError as e:
        raise ProviderError(f"Erreur de communication avec Ollama : {e}")


def embed(texts: list[str], task: str = "document") -> list[list[float]]:
    data = _post("/api/embed", {"model": config.EMBED_MODEL, "input": texts})
    return data["embeddings"]


def chat(messages: list[dict]) -> str:
    data = _post(
        "/api/chat",
        {
            "model": config.CHAT_MODEL,
            "messages": messages,
            "stream": False,
            "options": {"temperature": 0.2, "num_ctx": 4096},
        },
    )
    return data["message"]["content"]


def chat_stream(messages: list[dict]):
    """Génère la réponse morceau par morceau."""
    payload = {
        "model": config.CHAT_MODEL,
        "messages": messages,
        "stream": True,
        "options": {"temperature": 0.2, "num_ctx": 4096},
    }
    try:
        with _client.stream("POST", "/api/chat", json=payload) as r:
            if r.status_code != 200:
                r.read()
                raise ProviderError(f"Erreur Ollama {r.status_code} : {r.text}")
            for line in r.iter_lines():
                if not line:
                    continue
                data = json.loads(line)
                piece = data.get("message", {}).get("content", "")
                if piece:
                    yield piece
                if data.get("done"):
                    break
    except httpx.ConnectError:
        raise ProviderError("Ollama est injoignable. Est-il lancé ? (ollama serve)")
    except httpx.HTTPError as e:
        raise ProviderError(f"Erreur de communication avec Ollama : {e}")