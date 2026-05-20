from src.core.llm import llm_model
from src.common.logger import get_logger

logger = get_logger(__name__)

REWRITE_PROMPT = """
You are a medical query rewriter.

Rewrite the user's question into a concise, medically relevant search query 
optimized for retrieval in a Medical RAG system.

Rules:
- Preserve the original meaning
- Add relevant medical terminology if useful
- Keep the query short and clear
- Do not answer the question
- Return only the rewritten query

Examples:

Question: I have chest pain
Query: chest pain causes and treatment

Question: My head hurts and I feel dizzy
Query: headache with dizziness causes

Question: What tablet for fever?
Query: fever treatment and medications

Question: {question}

Rewritten Query:
"""


def rewrite_query(question: str) -> str:
    try:
        logger.info(f"Rewriting query: '{question}'")

        llm    = llm_model()
        prompt = REWRITE_PROMPT.format(question=question)

        response       = llm.invoke(prompt)
        rewritten      = response.content.strip()

        logger.info(f"Rewritten query: '{rewritten}'")
        return rewritten

    except Exception as e:
        logger.warning(f"Query rewrite failed, using original: {str(e)}")
        return question     # fallback to original