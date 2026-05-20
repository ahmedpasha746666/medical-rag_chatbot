"""
semantic_cache.py
──────────────────
Avoids repeated LLM calls by caching past Q&A pairs.

How it works:
    1. Every answered question is saved with its embedding vector
    2. When a new question comes in:
       - Embed the new question
       - Compare with all cached question embeddings (cosine similarity)
       - If similarity >= threshold → return cached answer  (cache HIT)
       - If not → return None                               (cache MISS)
    3. After LLM answers → call cache.store() to save for future use

Storage:
    data/processed/semantic_cache.json
    {
        "questions": ["what is diabetes?", ...],
        "answers":   ["Diabetes is ...",   ...],
        "embeddings": [[0.12, 0.34, ...],  ...]
    }
"""

import json
import numpy as np
from pathlib import Path

from src.core.llm import get_embedding_model
from src.common.logger import get_logger
from src.common.custom_exception import CustomException

logger = get_logger(__name__)

CACHE_FILE       = Path("data/processed/semantic_cache.json")
SIMILARITY_THRESHOLD = 0.85          # 0 to 1 — tune this as needed


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_cache() -> dict:
    """Load cache from disk. Returns empty cache if file doesn't exist."""
    if CACHE_FILE.exists():
        try:
            return json.loads(CACHE_FILE.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            logger.warning("Cache file corrupt — starting fresh.")
    return {"questions": [], "answers": [], "embeddings": []}


def _save_cache(cache: dict) -> None:
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    CACHE_FILE.write_text(
        json.dumps(cache, indent=2),
        encoding="utf-8"
    )


def _cosine_similarity(vec1: list, vec2: list) -> float:
    """Cosine similarity between two vectors. Returns value between 0 and 1."""
    a = np.array(vec1)
    b = np.array(vec2)
    # Avoid division by zero
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


# ── Main Class ────────────────────────────────────────────────────────────────

class SemanticCache:
    """
    Simple semantic cache for RAG Q&A.

    Usage:
        cache = SemanticCache()

        # Check before calling LLM
        cached_answer = cache.get(question)
        if cached_answer:
            return cached_answer        # cache HIT — no LLM call

        # Call LLM...
        answer = llm_response

        # Store for future
        cache.store(question, answer)
    """

    def __init__(self):
        self.embedding_model = get_embedding_model()
        logger.info("SemanticCache initialized.")

    def get(self, question: str) -> str | None:
        """
        Check if a similar question exists in cache.

        Returns cached answer if similarity >= threshold.
        Returns None if no similar question found (cache miss).
        """
        try:
            cache = _load_cache()

            # Nothing cached yet
            if not cache["questions"]:
                logger.info("Cache is empty — cache MISS.")
                return None

            # Embed the incoming question
            question_embedding = self.embedding_model.embed_query(question)

            # Compare with every cached question
            best_score = 0.0
            best_index = -1

            for i, cached_embedding in enumerate(cache["embeddings"]):
                score = _cosine_similarity(question_embedding, cached_embedding)
                if score > best_score:
                    best_score = score
                    best_index = i

            # Check if best match is above threshold
            if best_score >= SIMILARITY_THRESHOLD:
                cached_question = cache["questions"][best_index]
                cached_answer   = cache["answers"][best_index]
                logger.info(
                    f"Cache HIT (similarity={best_score:.2f}) | "
                    f"Matched: '{cached_question}'"
                )
                return cached_answer

            logger.info(f"Cache MISS (best similarity={best_score:.2f})")
            return None

        except Exception as e:
            logger.error(f"Cache get error: {str(e)}")
            return None       # on error, fall through to LLM

    def store(self, question: str, answer: str) -> None:
        """Save a new Q&A pair with its embedding to the cache."""
        try:
            cache = _load_cache()

            question_embedding = self.embedding_model.embed_query(question)

            cache["questions"].append(question)
            cache["answers"].append(answer)
            cache["embeddings"].append(question_embedding)

            _save_cache(cache)
            logger.info(f"Cache stored: '{question[:60]}...'")

        except Exception as e:
            logger.error(f"Cache store error: {str(e)}")
            raise CustomException(f"Cache store error: {str(e)}")

    def clear(self) -> None:
        """Delete all cached Q&A pairs."""
        _save_cache({"questions": [], "answers": [], "embeddings": []})
        logger.info("Semantic cache cleared.")

    def size(self) -> int:
        """Return number of cached Q&A pairs."""
        return len(_load_cache()["questions"])