# Submission Checklist — RAG Project

This file maps every requirement from the assignment instructions to where
it is satisfied in this project, for quick grading.

## 1. Required file structure ✅

| Required file | Present | Purpose |
|---|---|---|
| `01_documents.py` | ✅ | Loads raw documents (.txt / .pdf / .csv) |
| `02_preprocessing.py` | ✅ | Cleans and normalizes text |
| `03_chunking.py` | ✅ | Splits documents into overlapping chunks |
| `04_vector_representation.py` | ✅ | Generates embeddings (sentence-transformers) |
| `05_create_chroma_store.py` | ✅ | Builds and persists the Chroma vector store |
| `06_retrieve_context.py` | ✅ | Retrieves relevant chunks + builds labeled context |
| `07_prompting.py` | ✅ | Builds grounded prompt + calls OpenRouter API |
| `streamlit_app.py` | ✅ | Final deployed UI |
| `requirements.txt` | ✅ | All dependencies pinned |

## 2. Pipeline sequence followed ✅

```
documents -> preprocessing -> chunking -> vector representation
-> vector store -> context retrieval -> prompting -> Streamlit UI
```
Each numbered file is a self-contained stage and imports only the stage(s)
before it (see the `importlib.import_module(...)` calls at the top of each
file). `streamlit_app.py` only orchestrates stages 5 and 7; it contains no
pipeline logic of its own.

## 3. API key rules ✅

- No real API key is hard-coded anywhere in the Python files.
- `.env` (the file that would hold a real key locally) is excluded via
  `.gitignore` and was never included in this ZIP — only `.env.example`
  (a template with a placeholder) is included.
- `streamlit_app.py` reads the key from Streamlit Cloud secrets at deploy
  time, using the exact pattern required by the instructions:

```python
try:
    if not rag.OPENROUTER_API_KEY:
        rag.OPENROUTER_API_KEY = st.secrets.get("OPENROUTER_API_KEY", "")
        rag.OPENROUTER_MODEL = st.secrets.get("OPENROUTER_MODEL", rag.OPENROUTER_MODEL)
except Exception:
    pass
```

## 4. Answer quality requirements ✅

- **Uses retrieved context**: `07_prompting.py` builds a strict prompt that
  instructs the model to answer only from the provided context, and to say
  "Not enough information in the provided context." otherwise.
- **Cites sources**: `06_retrieve_context.py` labels every retrieved chunk
  as `[Source 1]`, `[Source 2]`, etc.; the prompt requires the model to
  reference these numbers; `streamlit_app.py` displays the matching source
  text underneath every answer.

## 5. What is included in this ZIP

- All 9 required Python files + `requirements.txt`
- `README.md` (Arabic + English) — full setup, usage, and deployment guide
- `data/documents/` — 3 ready-to-use sample text files so the project runs
  out of the box without needing any external dataset
- `.env.example`, `.streamlit/secrets.toml.example`, `.gitignore`

## 6. What the student must still do before final submission

These three steps require the student's own accounts/machine and cannot be
done inside this delivered ZIP:

1. **Run locally** to confirm it works (`pip install -r requirements.txt`,
   add a real key to `.env`, run `python 05_create_chroma_store.py`, then
   `streamlit run streamlit_app.py`).
2. **Push to a public GitHub repository** (the `.gitignore` already keeps
   the real key and the generated vector store out of the repo).
3. **Deploy on Streamlit Cloud**, adding `OPENROUTER_API_KEY` and
   `OPENROUTER_MODEL` under the app's Secrets panel.

Then submit all three required items: the ZIP file, the GitHub repo link,
and the deployed Streamlit app link.
