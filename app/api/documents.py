import json
import time
import uuid
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile

from app import providers
from app.services import vector_store
from app.services.chunking import chunk_document
from app.services.pdf_parser import extract_pages, looks_scanned

router = APIRouter(prefix="/documents", tags=["documents"])

BASE_DIR = Path(__file__).resolve().parents[2]  # dossier backend/
UPLOAD_DIR = BASE_DIR / "data" / "uploads"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

MAX_SIZE_MB = 50


@router.post("")
async def upload_document(file: UploadFile = File(...)):
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(400, "Seuls les fichiers PDF sont acceptés pour l'instant.")

    content = await file.read()
    if len(content) > MAX_SIZE_MB * 1024 * 1024:
        raise HTTPException(413, f"Fichier trop gros (max {MAX_SIZE_MB} Mo).")
    if not content.startswith(b"%PDF"):
        raise HTTPException(400, "Ce fichier n'est pas un PDF valide.")

    doc_id = uuid.uuid4().hex
    pdf_path = UPLOAD_DIR / f"{doc_id}.pdf"
    pdf_path.write_bytes(content)

    try:
        pages = extract_pages(str(pdf_path))
    except Exception as e:
        pdf_path.unlink(missing_ok=True)
        raise HTTPException(422, f"Impossible de lire le PDF : {e}")

    chunks = chunk_document(pages)
    scanned = looks_scanned(pages)

    meta = {
        "id": doc_id,
        "filename": file.filename,
        "page_count": len(pages),
        "chunk_count": len(chunks),
        "total_chars": sum(len(p["text"]) for p in pages),
        "looks_scanned": scanned,
    }
    (PROCESSED_DIR / f"{doc_id}.json").write_text(
        json.dumps({"meta": meta, "chunks": chunks}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if scanned:
        meta["warning"] = "Très peu de texte détecté : PDF probablement scanné (OCR nécessaire)."
    return meta


def _load(doc_id: str) -> dict:
    path = PROCESSED_DIR / f"{doc_id}.json"
    if not path.exists() or not doc_id.isalnum():
        raise HTTPException(404, "Document introuvable.")
    return json.loads(path.read_text(encoding="utf-8"))


@router.get("/{doc_id}")
def get_document(doc_id: str):
    return _load(doc_id)["meta"]


@router.get("/{doc_id}/chunks")
def get_chunks(doc_id: str, page: int | None = None, limit: int = 20):
    chunks = _load(doc_id)["chunks"]
    if page is not None:
        chunks = [c for c in chunks if c["page"] == page]
    return chunks[:limit]


INDEX_STATUS: dict[str, dict] = {}
BATCH = {"ollama": 16, "gemini": 10}


def _run_indexing(doc_id: str, provider_name: str) -> None:
    st = INDEX_STATUS[doc_id]
    try:
        provider = providers.get(provider_name)
        chunks = _load(doc_id)["chunks"]
        texts = [c["text"] for c in chunks]
        step = BATCH[provider_name]
        embeddings: list[list[float]] = []
        for i in range(0, len(texts), step):
            embeddings.extend(provider.embed(texts[i : i + step], task="document"))
            st["done"] = len(embeddings)
            if provider_name == "gemini":
                time.sleep(4)
        vector_store.index_chunks(
            doc_id,
            chunks,
            embeddings,
            {"provider": provider_name, "model": providers.embed_model(provider_name)},
        )
        st["state"] = "done"
    except Exception as e:  # noqa: BLE001
        st["state"] = "error"
        st["error"] = str(e)


@router.post("/{doc_id}/index", status_code=202)
def index_document(doc_id: str, background: BackgroundTasks, provider: str = "ollama"):
    try:
        providers.get(provider)
    except ValueError as e:
        raise HTTPException(400, str(e))

    chunks = _load(doc_id)["chunks"]
    if not chunks:
        raise HTTPException(422, "Aucun texte à indexer (PDF scanné ?).")

    if INDEX_STATUS.get(doc_id, {}).get("state") == "running":
        return INDEX_STATUS[doc_id]

    INDEX_STATUS[doc_id] = {"state": "running", "provider": provider, "done": 0, "total": len(chunks)}
    background.add_task(_run_indexing, doc_id, provider)
    return INDEX_STATUS[doc_id]


@router.get("/{doc_id}/status")
def index_status(doc_id: str):
    _load(doc_id)  # 404 si le document n'existe pas
    if doc_id in INDEX_STATUS:
        return INDEX_STATUS[doc_id]
    if vector_store.is_indexed(doc_id):
        return {"state": "done", **vector_store.get_meta(doc_id)}
    return {"state": "not_indexed"}