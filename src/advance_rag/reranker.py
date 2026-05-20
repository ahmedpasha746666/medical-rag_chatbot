from src.core.llm import llm_model
from src.common.logger import get_logger

logger = get_logger(__name__)

SCORE_PROMPT = """
You are a medical relevance judge.

Question: {question}

Document: {document}

Score how relevant this document is to the question.
Reply with ONLY a number between 0 and 1. Nothing else.
Example: 0.8
"""


def rerank(question: str, docs: list, top_k: int = 2) -> list:
    if not docs:
        return docs

    try:
        logger.info(f"Reranking {len(docs)} docs...")
        llm    = llm_model()
        scored = []

        for doc in docs:
            prompt = SCORE_PROMPT.format(
                question=question,
                document=doc.page_content[:500]   # limit to avoid token overflow
            )

            response = llm.invoke(prompt)

            # Parse score safely
            try:
                score = float(response.content.strip())
            except ValueError:
                score = 0.0    # if LLM returns unexpected text, score = 0

            scored.append((score, doc))
            logger.info(f"Doc scored: {score:.2f}")

        # Sort by score descending, return top_k
        scored.sort(key=lambda x: x[0], reverse=True)
        top_docs = [doc for _, doc in scored[:top_k]]

        logger.info(f"Reranking complete. Kept top {top_k} docs.")
        return top_docs

    except Exception as e:
        logger.warning(f"Reranking failed, using original docs: {str(e)}")
        return docs[:top_k]  