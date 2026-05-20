import os
import warnings

# ── Silence ALL warnings before any imports ───────────────────────────────────
os.environ["TRANSFORMERS_VERBOSITY"]        = "error"
os.environ["TOKENIZERS_PARALLELISM"]        = "false"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
warnings.filterwarnings("ignore")

import streamlit as st
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from src.utils.file_handler import save_multiple_files
from src.core.rag_pipeline import process_uploaded_pdfs
from src.core.pdf_loader import load_pdf
from src.core.chunking import split_text
from src.advance_rag.streaming import get_streaming_response
from src.config.config import vector_store_path
from src.advance_rag.memory import clear_memory
from src.advance_rag.semantic_cache import SemanticCache


# ── Page setup ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Medical RAG Chatbot", page_icon="🏥", layout="wide")
st.title("🏥 Medical RAG Chatbot")

# ── Session state ─────────────────────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "stats" not in st.session_state:
    st.session_state.stats = {"total_pdfs": 0, "total_chunks": 0}

if "pending_question" not in st.session_state:
    st.session_state.pending_question = None


# ── Load chunks ONCE using cache_resource (survives reruns, loads only once) ──
@st.cache_resource
def load_chunks():
    """Load and chunk all PDFs. Cached — runs only once per app session."""
    try:
        docs   = load_pdf()
        chunks = split_text(docs)
        return chunks
    except Exception:
        return []   # no PDFs yet


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.header("📂 Upload PDFs")

    uploaded = st.file_uploader(
        "Choose PDF files",
        type=["pdf"],
        accept_multiple_files=True,
    )

    if uploaded and st.button("🚀 Process PDFs"):

        status_box = st.empty()

        status_box.info("📥 Saving uploaded files...")
        saved_paths, errors = save_multiple_files(uploaded)

        for err in errors:
            st.warning(err)

        if saved_paths:
            result = process_uploaded_pdfs(status_box=status_box)

            if result["success"]:
                status_box.success("✅ Done! PDFs indexed successfully.")
                st.session_state.stats["total_pdfs"]  += len(saved_paths)
                st.session_state.stats["total_chunks"] = result["total_chunks"]

                # Clear cache so chunks reload with newly added PDFs
                load_chunks.clear()

            else:
                status_box.error(f"❌ Failed: {result['error']}")

    st.markdown("---")

    # ── Sidebar Stats ─────────────────────────────────────────────────────────
    st.header("📊 Stats")
    index_ready = (Path(vector_store_path) / "index.faiss").exists()

    st.markdown(f"**Total PDFs:** {st.session_state.stats['total_pdfs']}")
    st.markdown(f"**Chunks:** {st.session_state.stats['total_chunks']}")
    st.markdown(f"**Vector DB:** {'✅ Ready' if index_ready else '❌ Not built yet'}")
    st.markdown(f"**Cached Q&A:** {SemanticCache().size()}")

    st.markdown("---")

    # ── Clear Buttons ─────────────────────────────────────────────────────────
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.pending_question = None
        st.rerun()

    if st.button("🧠 Clear Memory", use_container_width=True):
        clear_memory()
        st.success("Memory cleared.")

# ══════════════════════════════════════════════════════════════════════════════
# MAIN – Chat
# ══════════════════════════════════════════════════════════════════════════════

if not index_ready:
    st.info("👈 Upload and process PDFs from the sidebar to get started.")
    st.stop()

# ── Render chat history ───────────────────────────────────────────────────────
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("📄 Sources"):
                for doc in msg["sources"]:
                    pdf_name = Path(doc.metadata.get("source", "Unknown")).name
                    page     = doc.metadata.get("page", "?")
                    st.markdown(f"**📑 {pdf_name}** — page {page}")
                    st.caption(doc.page_content[:300])
                    st.divider()

# ── Chat input ────────────────────────────────────────────────────────────────
question = st.chat_input("Ask a medical question...")

if question:
    st.session_state.pending_question = question
    st.session_state.chat_history.append({"role": "user", "content": question})
    st.rerun()

# ── Process pending question ──────────────────────────────────────────────────
if st.session_state.pending_question:
    question = st.session_state.pending_question
    st.session_state.pending_question = None

    with st.chat_message("assistant"):
        with st.spinner("🤔 Thinking... searching your documents"):
            answer, sources = get_streaming_response(
                question,
                chunks=load_chunks()    # cached — no reload
            )

        if sources:
            with st.expander("📄 Sources"):
                for doc in sources:
                    pdf_name = Path(doc.metadata.get("source", "Unknown")).name
                    page     = doc.metadata.get("page", "?")
                    st.markdown(f"**📑 {pdf_name}** — page {page}")
                    st.caption(doc.page_content[:300])
                    st.divider()

    st.session_state.chat_history.append({
        "role":    "assistant",
        "content": answer,
        "sources": sources,
    })