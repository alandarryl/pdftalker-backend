def chunk_page(text: str, page: int, size: int = 1200, overlap: int = 200) -> list[dict]:
    """Découpe le texte d'une page en morceaux d'environ `size` caractères,
    avec `overlap` caractères de chevauchement. Coupe de préférence sur un
    saut de ligne ou un espace pour ne pas casser les mots."""
    chunks = []
    start = 0
    n = len(text)

    while start < n:
        end = min(start + size, n)

        if end < n:
            # cherche une coupure propre dans la seconde moitié de la fenêtre
            window_start = start + size // 2
            cut = max(text.rfind("\n", window_start, end), text.rfind(". ", window_start, end))
            if cut == -1:
                cut = text.rfind(" ", window_start, end)
            if cut != -1:
                end = cut + 1

        piece = text[start:end].strip()
        if piece:
            chunks.append({"page": page, "text": piece})

        if end >= n:
            break
        start = max(end - overlap, start + 1)

    return chunks


def chunk_document(pages: list[dict], size: int = 1200, overlap: int = 200) -> list[dict]:
    all_chunks = []
    for p in pages:
        all_chunks.extend(chunk_page(p["text"], p["page"], size, overlap))
    for idx, c in enumerate(all_chunks):
        c["chunk_id"] = idx
    return all_chunks