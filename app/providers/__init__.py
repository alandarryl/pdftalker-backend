from app import config
from app.providers import gemini, ollama


def get(name: str):
    if name == "ollama":
        return ollama
    if name == "gemini":
        return gemini
    raise ValueError(f"Provider inconnu : {name} (choix : ollama, gemini)")


def embed_model(name: str) -> str:
    return config.EMBED_MODEL if name == "ollama" else config.GEMINI_EMBED_MODEL