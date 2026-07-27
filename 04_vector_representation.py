"""
04_vector_representation.py
-----------------------------
Stage 4 of the RAG pipeline: VECTOR REPRESENTATION.

Job of this file: turn text into numbers (embedding vectors) that capture
meaning. Two chunks that talk about the same idea end up with similar
vectors even if they don't use the exact same words - this is what makes
semantic search possible in Stage 6 (retrieval).

Model used: "all-MiniLM-L6-v2" from sentence-transformers. It's small,
fast, runs on CPU, and free (downloaded once from Hugging Face, then cached
locally) - a good default for a student project.

Run this file directly to embed the chunks and see the vector shape:
    python 04_vector_representation.py
"""

import importlib

documents_module = importlib.import_module("01_documents")
preprocessing_module = importlib.import_module("02_preprocessing")
chunking_module = importlib.import_module("03_chunking")

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

_embedding_model = None  # lazy-loaded singleton, so we only load it once per process


def get_embedding_model():
    """Load (once) and return the sentence-transformers model."""
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _embedding_model


def embed_texts(texts):
    """
    Turn a list of strings into a list of embedding vectors (list[list[float]]).
    Used both to embed the knowledge base (once, at index time) and to embed
    the user's question (every time, at query time).
    """
    if not texts:
        return []
    model = get_embedding_model()
    vectors = model.encode(list(texts), normalize_embeddings=True, show_progress_bar=False)
    return vectors.tolist()


def embed_chunks(chunks):
    """Add an 'embedding' field to every chunk dict, in place, and return it."""
    texts = [chunk["text"] for chunk in chunks]
    vectors = embed_texts(texts)
    for chunk, vector in zip(chunks, vectors):
        chunk["embedding"] = vector
    return chunks


if __name__ == "__main__":
    raw_documents = documents_module.load_documents()
    cleaned = preprocessing_module.preprocess_documents(raw_documents)
    chunks = chunking_module.chunk_documents(cleaned)
    embedded = embed_chunks(chunks)

    print(f"Embedded {len(embedded)} chunk(s) with '{EMBEDDING_MODEL_NAME}'.")
    print(f"Vector length: {len(embedded[0]['embedding'])}")
