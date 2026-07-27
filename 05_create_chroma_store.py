"""
05_create_chroma_store.py
---------------------------
Stage 5 of the RAG pipeline: VECTOR STORE.

Job of this file: save the embedded chunks into a Chroma vector database on
disk, so Stage 6 (retrieval) can search them instantly without re-embedding
everything on every app run.

This is the "indexing" script. Run it once whenever your source documents
change:
    python 05_create_chroma_store.py

It rebuilds data/chroma_store/ from scratch every time it runs, so it's
always safe to re-run.
"""

import importlib
import shutil
import os

documents_module = importlib.import_module("01_documents")
preprocessing_module = importlib.import_module("02_preprocessing")
chunking_module = importlib.import_module("03_chunking")
vector_module = importlib.import_module("04_vector_representation")

CHROMA_PERSIST_DIR = os.path.join("data", "chroma_store")
COLLECTION_NAME = "rag_knowledge_base"


def get_chroma_client(persist_directory=CHROMA_PERSIST_DIR):
    import chromadb
    return chromadb.PersistentClient(path=persist_directory)


def build_vector_store(chunks, persist_directory=CHROMA_PERSIST_DIR,
                        collection_name=COLLECTION_NAME, rebuild=True):
    """
    Embed `chunks` (if not already embedded) and write them into a
    persistent Chroma collection. Returns the collection object.
    """
    if rebuild and os.path.isdir(persist_directory):
        shutil.rmtree(persist_directory)
    os.makedirs(persist_directory, exist_ok=True)

    if "embedding" not in chunks[0]:
        chunks = vector_module.embed_chunks(chunks)

    client = get_chroma_client(persist_directory)
    collection = client.get_or_create_collection(name=collection_name)

    collection.add(
        ids=[chunk["chunk_id"] for chunk in chunks],
        embeddings=[chunk["embedding"] for chunk in chunks],
        documents=[chunk["text"] for chunk in chunks],
        metadatas=[
            {
                "doc_id": chunk["doc_id"],
                "source": chunk["source"],
                "title": chunk["title"],
            }
            for chunk in chunks
        ],
    )
    return collection


def get_collection(persist_directory=CHROMA_PERSIST_DIR, collection_name=COLLECTION_NAME):
    """Open an existing persisted collection (used at query time)."""
    if not os.path.isdir(persist_directory):
        raise FileNotFoundError(
            f"No vector store found at '{persist_directory}'. "
            "Run: python 05_create_chroma_store.py"
        )
    client = get_chroma_client(persist_directory)
    return client.get_collection(name=collection_name)


def run_full_indexing_pipeline():
    """Convenience function: documents -> preprocessing -> chunking -> vectors -> store."""
    raw_documents = documents_module.load_documents()
    cleaned = preprocessing_module.preprocess_documents(raw_documents)
    chunks = chunking_module.chunk_documents(cleaned)
    chunks = vector_module.embed_chunks(chunks)
    collection = build_vector_store(chunks)
    return collection, chunks


if __name__ == "__main__":
    collection, chunks = run_full_indexing_pipeline()
    print(f"Vector store built at '{CHROMA_PERSIST_DIR}'")
    print(f"Collection '{COLLECTION_NAME}' now has {collection.count()} chunk(s).")
