from pathlib import Path
from src.common.logger import get_logger
from src.common.custom_exception import CustomException

logger = get_logger(__name__)

UPLOAD_DIR = Path("data/uploads")


def save_uploaded_file(file_name: str, file_bytes: bytes) -> Path:
    if not file_name.lower().endswith(".pdf"):  # 1. Must be a PDF
        raise CustomException(f"'{file_name}' is not a PDF.")

    if len(file_bytes) == 0:   # 2. Must not be empty
        raise CustomException(f"'{file_name}' is empty.")

    if not file_bytes.startswith(b"%PDF-"): # 3. Check PDF magic bytes
        raise CustomException(f"'{file_name}' is not a valid PDF.")

    # 4. Save to disk
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    save_path = UPLOAD_DIR / file_name
    save_path.write_bytes(file_bytes)

    logger.info(f"Saved: {save_path}")
    return save_path


def save_multiple_files(uploaded_files: list) -> tuple[list, list]:
    """
    Save a list of Streamlit UploadedFile objects.
    Returns (saved_paths, errors).
    """
    saved_paths = []
    errors = []

    for uf in uploaded_files:
        try:
            path = save_uploaded_file(uf.name, uf.read())
            saved_paths.append(path)
        except CustomException as e:
            errors.append(f"❌ {e}")

    return saved_paths, errors