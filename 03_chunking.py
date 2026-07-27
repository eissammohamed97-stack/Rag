"""
03_chunking.py
---------------
Stage 3 of the RAG pipeline: CHUNKING.

Job of this file: split each cleaned document into smaller overlapping
pieces ("chunks"). Whole documents are usually too long and too broad to
embed well or to hand to an LLM - a chunk should be small enough to be
about ONE topic, so retrieval can pull back just the relevant part instead
of an entire document.

Strategy used here: word-based sliding window with overlap. It's simple,
has no external dependencies, and works well enough for plain text.

Run this file directly to see chunking in action:
    python 03_chunking.py
"""

import importlib

documents_module = importlib.import_module("01_documents")
preprocessing_module = importlib.import_module("02_preprocessing")

# Tune these two numbers to trade off context size vs. retrieval precision.
CHUNK_SIZE_WORDS = 120     # words per chunk
CHUNK_OVERLAP_WORDS = 20   # words shared between consecutive chunks


def chunk_text(text, chunk_size=CHUNK_SIZE_WORDS, overlap=CHUNK_OVERLAP_WORDS):
    """Split `text` into a list of overlapping word-based chunks."""
    words = text.split()
    if not words:
        return []

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks = []
    start = 0
    step = chunk_size - overlap
    while start < len(words):
        piece = words[start:start + chunk_size]
        chunks.append(" ".join(piece))
        if start + chunk_size >= len(words):
            break
        start += step

    return chunks


def chunk_documents(documents, chunk_size=CHUNK_SIZE_WORDS, overlap=CHUNK_OVERLAP_WORDS):
    """
    Turn a list of documents into a flat list of chunks:

        [{"chunk_id": "0-0", "doc_id": 0, "source": "...", "title": "...",
          "text": "..."}, ...]
    """
    all_chunks = []
    for doc in documents:
        pieces = chunk_text(doc["text"], chunk_size=chunk_size, overlap=overlap)
        for i, piece in enumerate(pieces):
            all_chunks.append({
                "chunk_id": f"{doc['doc_id']}-{i}",
                "doc_id": doc["doc_id"],
                "source": doc["source"],
                "title": doc["title"],
                "text": piece,
            })

    if not all_chunks:
        raise ValueError("Chunking produced 0 chunks - check the input documents.")

    return all_chunks


if __name__ == "__main__":
    raw_documents = documents_module.load_documents()
    cleaned = preprocessing_module.preprocess_documents(raw_documents)
    chunks = chunk_documents(cleaned)

    print(f"Created {len(chunks)} chunk(s) from {len(cleaned)} document(s).\n")
    for chunk in chunks[:5]:
        preview = chunk["text"][:70].replace("\n", " ")
        print(f"  [{chunk['chunk_id']}] {chunk['title']:<20} {preview}...")
