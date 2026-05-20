from langchain_huggingface import HuggingFaceEmbeddings
from src.config.config import groq_api_key
from src.common.custom_exception import CustomException
from src.common.logger import get_logger
from langchain_groq import ChatGroq

logger = get_logger(__name__)

# ── Module-level cache — loaded once, reused forever ─────────────────────────
_embedding_model = None
_llm_model       = None


def get_embedding_model():
    global _embedding_model

    if _embedding_model is None:
        try:
            logger.info("Initializing HuggingFace Embeddings model")
            _embedding_model = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            logger.info("HuggingFace Embeddings model initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing HuggingFace Embeddings model: {str(e)}")
            raise CustomException(f"Error initializing HuggingFace Embeddings model: {str(e)}")
    else:
        logger.info("Reusing cached HuggingFace Embeddings model")

    return _embedding_model


def llm_model(groq_api_key: str = groq_api_key):
    global _llm_model

    if _llm_model is None:
        try:
            logger.info("Initializing Groq LLM model")
            _llm_model = ChatGroq(
                api_key=groq_api_key,
                model="llama-3.3-70b-versatile",
                temperature=0.3,
                max_tokens=2048,
                streaming=True
            )
            logger.info("Groq LLM model initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing groq llm model: {str(e)}")
            raise CustomException(f"Error initializing groq llm model: {str(e)}")
    else:
        logger.info("Reusing cached Groq LLM model")

    return _llm_model