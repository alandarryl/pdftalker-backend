import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parents[1]  # dossier backend/
DATA_DIR = BASE_DIR / "data"

load_dotenv(BASE_DIR / ".env")

# Ollama
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
EMBED_MODEL = os.getenv("EMBED_MODEL", "bge-m3")
CHAT_MODEL = os.getenv("CHAT_MODEL", "qwen2.5:1.5b")

# Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_EMBED_MODEL = os.getenv("GEMINI_EMBED_MODEL", "gemini-embedding-001")
GEMINI_CHAT_MODEL = os.getenv("GEMINI_CHAT_MODEL", "gemini-2.5-flash")

# Provider de génération par défaut ("ollama" ou "gemini")
CHAT_PROVIDER = os.getenv("CHAT_PROVIDER", "ollama")