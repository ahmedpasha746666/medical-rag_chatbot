from src.common.logger import get_logger
from src.common.custom_exception import CustomException
from src.core.pdf_loader import load_pdf
from src.core.chunking import split_text
from src.core.vectore_store import save_vector_store

logger = get_logger(__name__)


def process_uploaded_pdfs(status_box=None) -> dict:

    result = {"success": False, "total_chunks": 0, "error": ""}

    def update(msg: str):
        logger.info(msg)
        if status_box:
            status_box.info(msg)

    try:
        # Step 1 – Load
        update("📄 Loading PDFs...")
        documents = load_pdf()

        # Step 2 – Chunk
        update("✂️  Creating chunks...")
        chunks = split_text(documents)

        # Step 3 – Embed + Save
        update("🧠 Creating embeddings & saving Vector DB...")
        save_vector_store(chunks)

        result["success"]      = True
        result["total_chunks"] = len(chunks)

    except CustomException as e:
        result["error"] = str(e)
        logger.error(f"Pipeline failed: {e}")

    return result