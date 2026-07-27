"""
06_retrieve_context.py
------------------------
Stage 6 of the RAG pipeline: CONTEXT RETRIEVAL.

Job of this file: given a user's question, search the Chroma vector store
for the most relevant chunks, then assemble them into one clean, labeled
context block that can be handed to the LLM in Stage 7 (prompting).

Every source is numbered ([Source 1], [Source 2], ...) so the model - and
the user - can see exactly which chunk backs which part of the answer.
This is what makes citations possible in the final Streamlit app.

Run this file directly for a quick manual test:
    python 06_retrieve_context.py
"""

import importlib

vector_module = importlib.import_module("04_vector_representation")
store_module = importlib.import_module("05_create_chroma_store")

TOP_K = 4                 # how many chunks to retrieve per question
MAX_CONTEXT_WORDS = 350   # word budget for the final context block


def retrieve(question, k=TOP_K, persist_directory=store_module.CHROMA_PERSIST_DIR):
    """
    Embed the question and search Chroma for the top-k most similar chunks.
    Returns a list of dicts: [{"text", "title", "source", "distance"}, ...]
    ordered from most to least relevant.
    """
    collection = store_module.get_collection(persist_directory=persist_directory)
    query_embedding = vector_module.embed_texts([question])[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
    )

    retrieved = []
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]
    for text, meta, distance in zip(documents, metadatas, distances):
        retrieved.append({
            "text": text,
            "title": meta["title"],
            "source": meta["source"],
            "distance": distance,
        })
    return retrieved


def build_context(retrieved_chunks, max_words=MAX_CONTEXT_WORDS):
    """
    Turn retrieved chunks into a single labeled context string, stopping
    once the word budget is used up. Deduplicates identical chunk text.
    """
    used_texts = set()
    lines = []
    total_words = 0
    sources_used = []

    for position, chunk in enumerate(retrieved_chunks, start=1):
        normalized = chunk["text"].strip().lower()
        if normalized in used_texts:
            continue

        word_count = len(chunk["text"].split())
        if total_words + word_count > max_words and lines:
            continue

        used_texts.add(normalized)
        lines.append(f"[Source {len(sources_used) + 1}] {chunk['title']} ({chunk['source']})")
        lines.append(chunk["text"])
        lines.append("")
        total_words += word_count
        sources_used.append(chunk)

    return {
        "context_text": "\n".join(lines).strip(),
        "sources_used": sources_used,
    }


def get_context_for_question(question, k=TOP_K, max_words=MAX_CONTEXT_WORDS):
    """One-call convenience wrapper: question -> retrieve -> build_context."""
    retrieved_chunks = retrieve(question, k=k)
    return build_context(retrieved_chunks, max_words=max_words)


if __name__ == "__main__":
    demo_question = "How many vacation days do employees get?"
    result = get_context_for_question(demo_question)
    print("Question:", demo_question)
    print()
    print(result["context_text"])
