"""
02_preprocessing.py
--------------------
Stage 2 of the RAG pipeline: PREPROCESSING.

Job of this file: turn messy raw text into clean, consistent text.
Real-world documents have extra whitespace, broken line breaks, and
sometimes empty content. Cleaning this up now means every later stage
(chunking, embedding, retrieval) works with predictable input.

Run this file directly to preprocess the documents loaded by 01_documents.py:
    python 02_preprocessing.py
"""

import re
import importlib

documents_module = importlib.import_module("01_documents")


def clean_text(text):
    """Collapse whitespace, strip control characters, and trim the ends."""
    if text is None:
        return ""
    text = str(text)
    text = text.replace("\r", " ").replace("\t", " ")
    # Collapse 2+ newlines into a single paragraph break
    text = re.sub(r"\n{2,}", "\n\n", text)
    # Collapse runs of spaces/tabs into a single space (but keep newlines)
    text = re.sub(r"[ \t]+", " ", text)
    # Trim trailing spaces on each line
    text = "\n".join(line.strip() for line in text.split("\n"))
    return text.strip()


def preprocess_documents(raw_documents):
    """
    Clean every document's text and drop documents that end up empty.
    Returns a new list; does not mutate the input.
    """
    cleaned_documents = []
    for doc in raw_documents:
        cleaned_text = clean_text(doc["text"])
        if not cleaned_text:
            # Nothing useful left after cleaning -> skip it
            continue
        new_doc = dict(doc)
        new_doc["text"] = cleaned_text
        cleaned_documents.append(new_doc)

    if not cleaned_documents:
        raise ValueError("All documents were empty after preprocessing.")

    return cleaned_documents


if __name__ == "__main__":
    raw_documents = documents_module.load_documents()
    cleaned = preprocess_documents(raw_documents)
    print(f"Preprocessed {len(cleaned)}/{len(raw_documents)} document(s).\n")
    for doc in cleaned:
        print(f"  [{doc['doc_id']}] {doc['source']:<25} {len(doc['text'])} chars")
