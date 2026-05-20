from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.common.custom_exception import CustomException
from src.common.logger import get_logger

logger = get_logger(__name__)
    

def split_text(document):
    try:
        logger.info("Splitting text into chunks")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(document)
        logger.info("Text split into chunks successfully")
        return chunks
    
    except Exception as e:
        logger.error(f"Error splitting text into chunks: {str(e)}")
        raise CustomException(f"Error splitting text into chunks: {str(e)}")
       
