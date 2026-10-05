import fitz  # PyMuPDF


def extract_pages(pdf_path: str) -> list[dict]:
    """Retourne [{'page': 1, 'text': '...'}, ...] (pages numérotées à partir de 1)."""
    pages = []
    with fitz.open(pdf_path) as doc:
        for i, page in enumerate(doc, start=1):
            text = page.get_text("text").strip()
            pages.append({"page": i, "text": text})
    return pages


def looks_scanned(pages: list[dict], min_chars_per_page: int = 30) -> bool:
    """Heuristique : très peu de texte extrait => probablement un PDF scanné (OCR requis)."""
    if not pages:
        return True
    total = sum(len(p["text"]) for p in pages)
    return total < min_chars_per_page * len(pages)