"""
01_documents.py
----------------
Stage 1 of the RAG pipeline: DOCUMENTS.

Job of this file: find raw source files and load their text into memory.
Nothing here does any cleaning, splitting, or embedding yet - that happens
in the later stages. Keeping this stage "dumb" makes it easy to plug in a
new data source later (a folder of PDFs, a CSV export, a database, etc.)
without touching the rest of the pipeline.

Supported file types out of the box:
    - .txt   -> read as plain text
    - .pdf   -> text extracted page by page (requires the `pypdf` package)
    - .csv   -> one row = one document (requires a "text" column, an
                optional "title" column is used if present)

Run this file directly to sanity-check what will be loaded:
    python 01_documents.py
"""

import os
import csv

# Folder that holds the raw source files. Students can drop their own
# .txt / .pdf / .csv files in here - no code changes required.
DOCUMENTS_DIR = os.path.join("data", "documents")


def _load_txt(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def _load_pdf(path):
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise ImportError(
            "Reading PDF files requires the 'pypdf' package. "
            "Install it with: pip install pypdf"
        ) from exc

    reader = PdfReader(path)
    pages_text = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages_text)


def _load_csv(path):
    """Each row becomes one document. Expects a 'text' column."""
    docs = []
    with open(path, "r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "text" not in reader.fieldnames:
            raise ValueError(
                f"CSV file '{path}' must contain a 'text' column. "
                f"Found columns: {reader.fieldnames}"
            )
        for i, row in enumerate(reader):
            title = row.get("title") or f"{os.path.basename(path)} - row {i}"
            docs.append({"title": title, "text": row.get("text", "")})
    return docs


def load_documents(documents_dir=DOCUMENTS_DIR):
    """
    Walk `documents_dir` and return a list of raw documents:

        [{"doc_id": 0, "source": "company_policy.txt",
          "title": "company_policy", "text": "..."}, ...]

    This is the ONLY function the rest of the pipeline needs from this file.
    """
    if not os.path.isdir(documents_dir):
        raise FileNotFoundError(
            f"Documents folder not found: '{documents_dir}'. "
            "Create it and add your .txt / .pdf / .csv files."
        )

    raw_documents = []
    for filename in sorted(os.listdir(documents_dir)):
        path = os.path.join(documents_dir, filename)
        if not os.path.isfile(path):
            continue

        extension = os.path.splitext(filename)[1].lower()

        if extension == ".txt":
            text = _load_txt(path)
            raw_documents.append({
                "source": filename,
                "title": os.path.splitext(filename)[0],
                "text": text,
            })

        elif extension == ".pdf":
            text = _load_pdf(path)
            raw_documents.append({
                "source": filename,
                "title": os.path.splitext(filename)[0],
                "text": text,
            })

        elif extension == ".csv":
            for row in _load_csv(path):
                raw_documents.append({
                    "source": filename,
                    "title": row["title"],
                    "text": row["text"],
                })

        else:
            # Skip files we don't know how to read (.gitkeep, .DS_Store, ...)
            continue

    # Assign a stable numeric ID to every document, in load order
    for doc_id, doc in enumerate(raw_documents):
        doc["doc_id"] = doc_id

    if not raw_documents:
        raise ValueError(
            f"No readable documents found in '{documents_dir}'. "
            "Add at least one .txt, .pdf, or .csv file."
        )

    return raw_documents


if __name__ == "__main__":
    documents = load_documents()
    print(f"Loaded {len(documents)} document(s) from '{DOCUMENTS_DIR}':\n")
    for doc in documents:
        preview = doc["text"][:80].replace("\n", " ")
        print(f"  [{doc['doc_id']}] {doc['source']:<25} {preview}...")
