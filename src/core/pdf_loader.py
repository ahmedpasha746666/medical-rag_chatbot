from langchain_community.document_loaders import PyPDFLoader
from pathlib import Path
from src.common.custom_exception import CustomException
from src.common.logger import get_logger
from src.config.config import data_path

logger = get_logger(__name__)


def load_pdf():
    """Load PDF documents from directory, skipping problematic ones with encoding errors."""
    try:
        logger.info("Loading PDF documents")
        documents = []
        pdf_dir = Path(data_path)
        pdf_files = list(pdf_dir.rglob("*.pdf"))
        
        if not pdf_files:
            logger.warning(f"No PDF files found in {data_path}")
            return documents
        
        logger.info(f"Found {len(pdf_files)} PDF files to process")
        
        for pdf_file in pdf_files:
            try:
                logger.info(f"Loading: {pdf_file.name}")
                loader = PyPDFLoader(str(pdf_file))
                docs = loader.load()
                documents.extend(docs)
                logger.info(f"Successfully loaded {len(docs)} pages from {pdf_file.name}")
            except LookupError as e:
                if "encoding" in str(e).lower():
                    logger.warning(f"Skipping {pdf_file.name}: Encoding error - {str(e)}. This PDF may have unsupported fonts.")
                else:
                    logger.warning(f"Skipping {pdf_file.name}: {str(e)}")
            except Exception as e:
                logger.warning(f"Skipping {pdf_file.name}: {type(e).__name__} - {str(e)}")
        
        if not documents:
            raise CustomException("No documents could be loaded from any PDF files. All files had errors.")
        
        logger.info(f"PDF documents loaded successfully. Total pages: {len(documents)}")
        return documents
    
    except Exception as e:
        logger.error(f"Error loading PDF documents: {str(e)}")
        raise CustomException(f"Error loading PDF documents: {str(e)}")
    

       
