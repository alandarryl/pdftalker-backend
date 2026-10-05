import json

import numpy as np

from app import config

VEC_DIR = config.DATA_DIR / "vectors"
PROCESSED_DIR = config.DATA_DIR / "processed"
VEC_DIR.mkdir(parents=True, exist_ok=True)


def _normalize(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=-1, keepdims=True)
    norms[norms == 0] = 1
    return m / norms


def is_indexed(doc_id: str) -> bool:
    return (VEC_DIR / f"{doc_id}.npy").exists()


def get_meta(doc_id: str) -> dict:
    """Provider et modèle de l'index. Les index créés avant cette version n'ont
    pas de fichier meta : on suppose Ollama."""
    path = VEC_DIR / f"{doc_id}.meta.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"provider": "ollama", "model": config.EMBED_MODEL}


def index_chunks(doc_id: str, chunks: list[dict], embeddings: list[list[float]], meta: dict) -> None:
    if len(chunks) != len(embeddings):
        raise ValueError("Nombre d'embeddings différent du nombre de chunks.")
    mat = _normalize(np.asarray(embeddings, dtype=np.float32))
    np.save(VEC_DIR / f"{doc_id}.npy", mat)
    (VEC_DIR / f"{doc_id}.meta.json").write_text(
        json.dumps({**meta, "dim": int(mat.shape[1])}), encoding="utf-8"
    )


def search(doc_id: str, query_embedding: list[float], k: int = 4) -> list[dict]:
    path = VEC_DIR / f"{doc_id}.npy"
    if not path.exists():
        raise LookupError("Document non indexé.")

    mat = np.load(path)
    chunks = json.loads((PROCESSED_DIR / f"{doc_id}.json").read_text(encoding="utf-8"))["chunks"]
    if len(chunks) != len(mat):
        raise LookupError("Index obsolète : relance l'indexation.")

    q = _normalize(np.asarray(query_embedding, dtype=np.float32))
    if q.shape[0] != mat.shape[1]:
        raise LookupError("Index créé avec un autre modèle d'embedding : relance l'indexation.")
    scores = mat @ q
    top = np.argsort(-scores)[:k]
    return [
        {"text": chunks[i]["text"], "page": chunks[i]["page"], "score": round(float(scores[i]), 3)}
        for i in top
    ]