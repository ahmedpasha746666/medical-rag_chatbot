"""
memory.py
──────────
Stores conversation history so the LLM understands follow-up questions.

Example:
    User: "What is diabetes?"
    User: "What are its symptoms?"   ← "its" needs context to make sense

    Memory passes:
        [
            {"role": "user",      "content": "What is diabetes?"},
            {"role": "assistant", "content": "Diabetes is ..."},
            {"role": "user",      "content": "What are its symptoms?"}
        ]
    to the LLM so it understands "its" = diabetes.

Storage:
    data/chat_history/memory.json
"""

import json
from pathlib import Path
from src.common.logger import get_logger

logger = get_logger(__name__)

MEMORY_FILE = Path("data/chat_history/memory.json")
MAX_TURNS   = 10    # keep last 10 exchanges to avoid token overflow


def _load() -> list:
    if MEMORY_FILE.exists():
        try:
            return json.loads(MEMORY_FILE.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return []
    return []


def _save(history: list) -> None:
    MEMORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    MEMORY_FILE.write_text(json.dumps(history, indent=2), encoding="utf-8")


def add_to_memory(question: str, answer: str) -> None:
    """Save a Q&A pair to memory."""
    history = _load()
    history.append({"role": "user",      "content": question})
    history.append({"role": "assistant", "content": answer})

    # Keep only last MAX_TURNS exchanges (each turn = 2 messages)
    history = history[-(MAX_TURNS * 2):]

    _save(history)
    logger.info(f"Memory updated. Total messages: {len(history)}")


def get_memory() -> list:
    """Return full chat history as list of role/content dicts."""
    return _load()


def format_memory_as_text() -> str:
    """
    Format chat history as plain text to inject into prompt.

    Returns empty string if no history.
    """
    history = _load()
    if not history:
        return ""

    lines = []
    for msg in history:
        role = "User" if msg["role"] == "user" else "Assistant"
        lines.append(f"{role}: {msg['content']}")

    return "\n".join(lines)


def clear_memory() -> None:
    """Wipe all chat history."""
    _save([])
    logger.info("Memory cleared.")