import time

import httpx

import json

from app import config
from app.providers.errors import ProviderError

_client = httpx.Client(base_url="https://generativelanguage.googleapis.com/v1beta", timeout=120)
EMBED_DIM = 768  # gemini-embedding-001 supporte plusieurs tailles ; 768 suffit et pèse peu


def _post(path: str, payload: dict) -> dict:
    if not config.GEMINI_API_KEY:
        raise ProviderError("GEMINI_API_KEY manquante dans le fichier .env (puis redémarre le serveur).")

    last = ""
    for attempt in range(4):
        try:
            r = _client.post(path, json=payload, headers={"x-goog-api-key": config.GEMINI_API_KEY})
        except httpx.HTTPError as e:
            raise ProviderError(f"Erreur de communication avec Gemini : {e}")

        if r.status_code == 200:
            return r.json()
        last = f"{r.status_code} : {r.text[:300]}"
        if r.status_code in (429, 500, 503):  # quota ou surcharge : on réessaie
            time.sleep(2 ** (attempt + 1))
            continue
        break
    raise ProviderError(f"Erreur Gemini {last}")


def embed(texts: list[str], task: str = "document") -> list[list[float]]:
    task_type = "RETRIEVAL_QUERY" if task == "query" else "RETRIEVAL_DOCUMENT"
    model = config.GEMINI_EMBED_MODEL
    out: list[list[float]] = []
    for i in range(0, len(texts), 50):
        batch = texts[i : i + 50]
        data = _post(
            f"/models/{model}:batchEmbedContents",
            {
                "requests": [
                    {
                        "model": f"models/{model}",
                        "content": {"parts": [{"text": t}]},
                        "taskType": task_type,
                        "outputDimensionality": EMBED_DIM,
                    }
                    for t in batch
                ]
            },
        )
        out.extend(e["values"] for e in data["embeddings"])
    return out


def chat(messages: list[dict]) -> str:
    system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
    contents = [
        {"role": "model" if m["role"] == "assistant" else "user", "parts": [{"text": m["content"]}]}
        for m in messages
        if m["role"] != "system"
    ]
    payload = {"contents": contents, "generationConfig": {"temperature": 0.2}}
    if system:
        payload["systemInstruction"] = {"parts": [{"text": system}]}

    data = _post(f"/models/{config.GEMINI_CHAT_MODEL}:generateContent", payload)
    try:
        return "".join(p.get("text", "") for p in data["candidates"][0]["content"]["parts"])
    except (KeyError, IndexError):
        raise ProviderError(f"Réponse Gemini vide ou bloquée : {str(data)[:300]}")



def chat_stream(messages: list[dict]):
    if not config.GEMINI_API_KEY:
        raise ProviderError("GEMINI_API_KEY manquante dans le fichier .env.")

    system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
    contents = [
        {"role": "model" if m["role"] == "assistant" else "user", "parts": [{"text": m["content"]}]}
        for m in messages
        if m["role"] != "system"
    ]
    payload = {"contents": contents, "generationConfig": {"temperature": 0.2}}
    if system:
        payload["systemInstruction"] = {"parts": [{"text": system}]}

    last = ""
    for attempt in range(4):
        started = False
        try:
            with _client.stream(
                "POST",
                f"/models/{config.GEMINI_CHAT_MODEL}:streamGenerateContent",
                params={"alt": "sse"},
                json=payload,
                headers={"x-goog-api-key": config.GEMINI_API_KEY},
            ) as r:
                if r.status_code != 200:
                    r.read()
                    last = f"{r.status_code} : {r.text[:300]}"
                    if r.status_code in (429, 500, 503) and attempt < 3:
                        time.sleep(2 ** (attempt + 1))
                        continue
                    raise ProviderError(f"Erreur Gemini {last}")
                for line in r.iter_lines():
                    if not line.startswith("data:"):
                        continue
                    raw = line[5:].strip()
                    if not raw:
                        continue
                    data = json.loads(raw)
                    try:
                        parts = data["candidates"][0]["content"]["parts"]
                    except (KeyError, IndexError):
                        continue
                    text = "".join(p.get("text", "") for p in parts)
                    if text:
                        started = True
                        yield text
                return
        except httpx.HTTPError as e:
            if started:
                raise ProviderError(f"Flux interrompu : {e}")
            raise ProviderError(f"Erreur de communication avec Gemini : {e}")