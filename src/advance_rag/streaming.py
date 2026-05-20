"""
streaming.py
─────────────
Full advanced RAG pipeline with streaming.

Flow:
    question
        ↓
    guardrails()           →  block non-medical questions
        ↓
    SemanticCache.get()    →  HIT → return cached answer instantly
        ↓ MISS
    rewrite_query()        →  clearer question for better retrieval
        ↓
    hybrid_search()        →  FAISS + BM25 combined retrieval
        ↓
    rerank()               →  keep top 2 most relevant docs
        ↓
    memory.format()        →  inject past conversation into prompt
        ↓
    llm.stream()           →  tokens appear one by one (typing effect)
        ↓
    memory.add()           →  save Q&A to memory
    cache.store()          →  save to semantic cache
"""

import streamlit as st
from langchain_core.prompts import PromptTemplate

from src.core.llm import llm_model
from src.advance_rag.semantic_cache import SemanticCache
from src.advance_rag.query_rewriter import rewrite_query
from src.advance_rag.reranker import rerank
from src.advance_rag.memory import add_to_memory, format_memory_as_text
from src.advance_rag.guardrals import is_medical_question
from src.advance_rag.hybrid_search import hybrid_search
from src.common.logger import get_logger
from src.common.custom_exception import CustomException

logger = get_logger(__name__)

PROMPT_TEMPLATE = """
You are an expert medical AI assistant. A doctor or patient is asking you a question
based on uploaded medical documents.

Instructions:
- Answer ONLY from the provided context.
- Be clear, structured, and medically accurate.
- Use bullet points or numbered steps when listing information.
- If the answer is not in the context, say:
  "I could not find this information in the uploaded documents."
- Do NOT make up information.

Previous conversation:
{history}

Context:
{context}

Question:
{question}

Answer:
"""

cache = SemanticCache()


def get_streaming_response(question: str, chunks: list) -> tuple[str, list]:
    """
    Full pipeline: guardrails → cache → rewrite → hybrid search
                   → rerank → memory → stream → store.

    Parameters
    ----------
    question : user question
    chunks   : pre-loaded document chunks (passed from Streamlit session state)
               avoids reloading PDFs on every question

    Returns (answer, sources).
    """
    try:
        # ── Step 1: Guardrails ────────────────────────────────────────────────
        allowed, reason = is_medical_question(question)
        if not allowed:
            msg = f"⚠️ I can only answer medical questions. {reason}"
            st.warning(msg)
            return msg, []

        # ── Step 2: Semantic cache check ──────────────────────────────────────
        cached_answer = cache.get(question)
        if cached_answer:
            st.info("⚡ Answered from cache (similar question was asked before)")
            st.markdown(cached_answer)
            return cached_answer, []

        # ── Step 3: Rewrite query ─────────────────────────────────────────────
        rewritten_question = rewrite_query(question)
        if rewritten_question != question:
            st.caption(f"💡 I am searching for: *{rewritten_question}*")

        # ── Step 4: Hybrid search — uses pre-loaded chunks ───────────────────
        # chunks passed in from session state — no PDF reload needed ✅
        logger.info("Running hybrid search...")
        docs = hybrid_search(rewritten_question, chunks, top_k=3)

        if not docs:
            msg = "I could not find this information in the uploaded documents."
            st.markdown(msg)
            return msg, []

        # ── Step 5: Rerank ────────────────────────────────────────────────────
        docs = rerank(question, docs, top_k=2)

        # ── Step 6: Build prompt with memory ─────────────────────────────────
        context      = "\n\n".join([doc.page_content for doc in docs])
        history      = format_memory_as_text()
        prompt       = PromptTemplate(
            template=PROMPT_TEMPLATE,
            input_variables=["history", "context", "question"]
        )
        final_prompt = prompt.format(
            history=history,
            context=context,
            question=question
        )

        # ── Step 7: Stream tokens ─────────────────────────────────────────────
        logger.info("Streaming LLM response...")
        llm    = llm_model()
        answer = ""

        def token_generator():
            nonlocal answer
            for chunk in llm.stream(final_prompt):
                token   = chunk.content
                answer += token
                yield token

        st.write_stream(token_generator())

        # ── Step 8: Save to memory + cache ───────────────────────────────────
        add_to_memory(question, answer)
        cache.store(question, answer)
        logger.info("Streaming complete. Saved to memory and cache.")

        return answer, docs

    except Exception as e:
        logger.error(f"Streaming error: {str(e)}")
        raise CustomException(f"Streaming error: {str(e)}")