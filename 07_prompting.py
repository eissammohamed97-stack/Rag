"""
07_prompting.py
-----------------
Stage 7 of the RAG pipeline: PROMPTING.

Job of this file:
    1. Build a "strict, grounded" prompt that forces the model to answer
       ONLY from the retrieved context and to cite its sources.
    2. Send that prompt to an LLM through the OpenRouter API and return
       the answer.
    3. Tie stages 1-7 together with one function, `answer_question`, that
       the Streamlit app (and anything else) can call directly.

API KEY RULES (see project instructions):
    - Never hard-code your real key in this file.
    - Locally: put OPENROUTER_API_KEY in a .env file (see .env.example)
      and it will be picked up automatically via python-dotenv.
    - On Streamlit Cloud: the key is read from Streamlit secrets instead
      (streamlit_app.py overwrites OPENROUTER_API_KEY / OPENROUTER_MODEL
      below after importing this module - see that file).

Run this file directly for a quick manual test (requires a real key in .env):
    python 07_prompting.py
"""

import os
import importlib
import requests

try:
    from dotenv import load_dotenv
    load_dotenv()  # reads a local .env file into the environment, if present
except ImportError:
    pass  # python-dotenv is optional locally; not needed at all on Streamlit Cloud

retrieve_module = importlib.import_module("06_retrieve_context")

# --- Configuration -----------------------------------------------------
# Read from environment variables locally. On Streamlit Cloud, the app
# overwrites these two values from st.secrets right after importing this
# module (see streamlit_app.py).
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "openai/gpt-4o-mini")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


def build_prompt(question, context_text):
    """
    A 'strict' grounded prompt: only use the given context, say clearly
    when the answer isn't in it, and always cite source numbers. This is
    the single biggest lever for reducing hallucination in a RAG system.
    """
    prompt = (
        "You are a grounded RAG assistant.\n\n"
        "Rules:\n"
        "1. Use only the information in the provided context. Never add outside knowledge.\n"
        "2. If the context does not clearly contain the answer, say exactly:\n"
        "   'Not enough information in the provided context.'\n"
        "3. Always cite the source numbers you used, like [Source 1], [Source 2].\n"
        "4. Keep the answer concise: one to three sentences.\n\n"
        f"Question: {question}\n\n"
        f"Context:\n{context_text}\n\n"
        "Answer (with citations):"
    )
    return prompt


def ask_openrouter(prompt, model=None, temperature=0.2, timeout=60):
    """Send `prompt` to an OpenRouter chat model and return the answer text."""
    model = model or OPENROUTER_MODEL

    if not OPENROUTER_API_KEY:
        return (
            "[No OPENROUTER_API_KEY configured. Add it to a local .env file, "
            "or to Streamlit secrets when deployed.]"
        )

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
    }

    try:
        response = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=timeout)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()
    except requests.exceptions.RequestException as exc:
        return f"[Error calling OpenRouter: {exc}]"
    except (KeyError, IndexError):
        return "[Unexpected response format from OpenRouter.]"


def answer_question(question, k=None):
    """
    Full pipeline in one call: retrieve context -> build prompt -> ask the
    model -> return everything the UI needs to display an answer with
    citations.
    """
    kwargs = {"k": k} if k else {}
    context_result = retrieve_module.get_context_for_question(question, **kwargs)

    if not context_result["sources_used"]:
        return {
            "question": question,
            "answer": "Not enough information in the provided context.",
            "context_text": "",
            "sources_used": [],
            "prompt": "",
        }

    prompt = build_prompt(question, context_result["context_text"])
    answer = ask_openrouter(prompt)

    return {
        "question": question,
        "answer": answer,
        "context_text": context_result["context_text"],
        "sources_used": context_result["sources_used"],
        "prompt": prompt,
    }


if __name__ == "__main__":
    demo_question = "How many vacation days do employees get?"
    result = answer_question(demo_question)
    print("QUESTION:", result["question"])
    print("ANSWER:  ", result["answer"])
    print()
    print("Sources used:")
    for src in result["sources_used"]:
        print(" -", src["title"], f"({src['source']})")
