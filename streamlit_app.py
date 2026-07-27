"""
streamlit_app.py
------------------
Final stage of the RAG pipeline: STREAMLIT UI.

This is the app users actually interact with. It does NOT reimplement any
pipeline logic - it just imports the numbered stage files and calls them,
exactly like the notebook / instructions describe:

    documents -> preprocessing -> chunking -> vector representation
    -> vector store -> context retrieval -> prompting -> Streamlit UI

Run locally:
    streamlit run streamlit_app.py

Deploy on Streamlit Cloud:
    1. Push this whole project to a public GitHub repo (WITHOUT your .env file).
    2. Create a new app on https://share.streamlit.io pointing at streamlit_app.py.
    3. In the app's "Manage app" -> "Secrets" panel, add:
           OPENROUTER_API_KEY = "your_openrouter_key_here"
           OPENROUTER_MODEL = "openai/gpt-4o-mini"
"""

import importlib
import os

import streamlit as st

# --- Import the numbered pipeline stages (filenames start with digits,
#     so they can't be imported with a normal `import 07_prompting`) -----
store_module = importlib.import_module("05_create_chroma_store")
rag = importlib.import_module("07_prompting")  # holds OPENROUTER_API_KEY / answer_question

# --- Read the API key from Streamlit secrets when deployed --------------
# Locally, 07_prompting.py already loaded it from a .env file. On
# Streamlit Cloud there is no .env file, so we fall back to st.secrets.
try:
    if not rag.OPENROUTER_API_KEY:
        rag.OPENROUTER_API_KEY = st.secrets.get("OPENROUTER_API_KEY", "")
        rag.OPENROUTER_MODEL = st.secrets.get("OPENROUTER_MODEL", rag.OPENROUTER_MODEL)
except Exception:
    pass


# --- Page setup -----------------------------------------------------------
st.set_page_config(page_title="RAG Assistant", page_icon="📚", layout="centered")

st.title("📚 RAG Assistant")
st.caption(
    "Ask a question and get an answer grounded in your own documents, "
    "with citations back to the exact source."
)

# --- Sidebar: status + admin controls -------------------------------------
with st.sidebar:
    st.header("⚙️ Status")

    key_status = "✅ configured" if rag.OPENROUTER_API_KEY else "❌ missing"
    st.write(f"**API key:** {key_status}")
    st.write(f"**Model:** `{rag.OPENROUTER_MODEL}`")

    store_exists = os.path.isdir(store_module.CHROMA_PERSIST_DIR)
    st.write(f"**Vector store:** {'✅ built' if store_exists else '❌ not built yet'}")

    st.divider()
    st.subheader("Knowledge base")
    st.write(
        "Add `.txt`, `.pdf`, or `.csv` files to `data/documents/`, "
        "then rebuild the index below."
    )
    if st.button("🔄 Rebuild vector store", use_container_width=True):
        with st.spinner("Running documents → preprocessing → chunking → embeddings → store..."):
            try:
                collection, chunks = store_module.run_full_indexing_pipeline()
                st.success(f"Indexed {len(chunks)} chunk(s) into the vector store.")
            except Exception as exc:
                st.error(f"Indexing failed: {exc}")

    st.divider()
    top_k = st.slider("Chunks to retrieve (k)", min_value=1, max_value=8, value=4)


# --- Make sure a vector store exists before accepting questions -----------
if not os.path.isdir(store_module.CHROMA_PERSIST_DIR):
    st.warning(
        "No vector store found yet. Click **🔄 Rebuild vector store** "
        "in the sidebar first."
    )
    st.stop()

# --- Main Q&A area ----------------------------------------------------------
question = st.text_input("Your question", placeholder="e.g. How many vacation days do employees get?")
ask_clicked = st.button("Ask", type="primary")

if ask_clicked and question.strip():
    with st.spinner("Retrieving context and asking the model..."):
        try:
            result = rag.answer_question(question, k=top_k)
        except Exception as exc:
            st.error(f"Something went wrong: {exc}")
            result = None

    if result:
        st.subheader("Answer")
        st.write(result["answer"])

        if result["sources_used"]:
            st.subheader("Sources")
            for i, src in enumerate(result["sources_used"], start=1):
                with st.expander(f"[Source {i}] {src['title']}  ·  {src['source']}"):
                    st.write(src["text"])
        else:
            st.info("No relevant sources were found for this question.")

elif ask_clicked:
    st.warning("Please type a question first.")
