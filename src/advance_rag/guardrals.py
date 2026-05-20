"""
guardrails.py
──────────────
Blocks non-medical questions before they reach the LLM.

Two layers:
    1. Allow greetings instantly
    2. Keyword block — fast, no LLM call
    3. LLM check — for borderline questions
"""

from src.core.llm import llm_model
from src.common.logger import get_logger

logger = get_logger(__name__)

# Greetings — always allow these through
GREETINGS = ["good morning", "good evening", "good afternoon",
             "hello", "hi", "hey", "good night", "thanks", "thank you"]

# Obvious non-medical topics — block immediately
BLOCKED_KEYWORDS = [
    "poem", "joke", "story", "stock", "crypto", "movie",
    "recipe", "weather", "sports", "politics", "game",
    "song", "lyrics", "code", "programming",
]

GUARDRAIL_PROMPT = """
You are a medical chatbot gatekeeper.

Is the following question related to medicine, health, or medical documents?
Reply with ONLY "yes" or "no".

Question: {question}

Answer:
"""


def is_medical_question(question: str) -> tuple[bool, str]:
    """
    Check if question is allowed.

    Returns
    -------
    (True,  "")          – question is allowed
    (False, "<reason>")  – question is blocked
    """
    question_lower = question.lower().strip()

    # Layer 1 — always allow greetings
    if any(question_lower.startswith(g) for g in GREETINGS):
        logger.info("Greeting detected — allowed.")
        return True, ""

    # Layer 2 — fast keyword block
    for keyword in BLOCKED_KEYWORDS:
        if keyword in question_lower:
            reason = f"'{keyword}' is not a medical topic."
            logger.info(f"Guardrail blocked (keyword): {question}")
            return False, reason

    # Layer 3 — LLM check for borderline questions
    try:
        llm      = llm_model()
        prompt   = GUARDRAIL_PROMPT.format(question=question)
        response = llm.invoke(prompt)
        answer   = response.content.strip().lower()

        if answer.startswith("no"):
            logger.info(f"Guardrail blocked (LLM): {question}")
            return False, "This question does not appear to be medically related."

    except Exception as e:
        logger.warning(f"Guardrail LLM check failed, allowing: {str(e)}")

    return True, ""