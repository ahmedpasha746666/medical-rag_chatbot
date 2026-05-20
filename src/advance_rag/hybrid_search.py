"""
hybrid_search.py
─────────────────
Combines FAISS (semantic) + BM25 (keyword) search for better retrieval.

Why?
    FAISS  → good at meaning  e.g. "heart attack" matches "myocardial infarction"
    BM25   → good at keywords e.g. exact drug names, medical codes

    Combined → better than either alone.

Install:
    pip install rank_bm25
"""

from langchain_community.retrievers import BM25Retriever
from src.core.vectore_store import load_vectorstore
from src.core.llm import get_embedding_model
from src.common.logger import get_logger

logger = get_logger(__name__)


def hybrid_search(question: str, docs_for_bm25: list, top_k: int = 3) -> list:
    """
    Run FAISS + BM25 search and merge results.

    Parameters
    ----------
    question      : user question
    docs_for_bm25 : all documents (chunks) to build BM25 index from
                    pass the chunks from your vector store or load_pdf()
    top_k         : how many docs to return from each search

    Returns
    -------
    merged list of unique docs (FAISS results + BM25 results, deduplicated)
    """
    merged = []
    seen   = set()   # track content to avoid duplicates

    # ── FAISS semantic search ─────────────────────────────────────────────────
    try:
        logger.info("Running FAISS semantic search...")
        db           = load_vectorstore()
        faiss_docs   = db.similarity_search(question, k=top_k)

        for doc in faiss_docs:
            if doc.page_content not in seen:
                merged.append(doc)
                seen.add(doc.page_content)

        logger.info(f"FAISS returned {len(faiss_docs)} docs.")

    except Exception as e:
        logger.warning(f"FAISS search failed: {str(e)}")

    # ── BM25 keyword search ───────────────────────────────────────────────────
    try:
        logger.info("Running BM25 keyword search...")
        bm25_retriever = BM25Retriever.from_documents(docs_for_bm25)
        bm25_retriever.k = top_k
        bm25_docs = bm25_retriever.invoke(question)

        for doc in bm25_docs:
            if doc.page_content not in seen:
                merged.append(doc)
                seen.add(doc.page_content)

        logger.info(f"BM25 returned {len(bm25_docs)} docs.")

    except Exception as e:
        logger.warning(f"BM25 search failed: {str(e)}")

    logger.info(f"Hybrid search total unique docs: {len(merged)}")
    return merged