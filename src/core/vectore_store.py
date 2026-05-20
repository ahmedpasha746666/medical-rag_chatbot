from langchain_community.vectorstores import FAISS
from src.core.llm import get_embedding_model
from src.config.config import vector_store_path
from src.core.chunking import split_text
from src.common.custom_exception import CustomException
from src.common.logger import get_logger


logger = get_logger(__name__)

def save_vector_store(text_chunks):
    try:
        if not text_chunks:
            raise CustomException("No chunks were found..")
        
        logger.info("Generating your new vectorstore")

        embedding_model = get_embedding_model()

        db = FAISS.from_documents(text_chunks,embedding_model)

        logger.info("Saving vectorstoree")

        db.save_local(vector_store_path)

        logger.info("Vectostore saved sucesfulyy...")

        return db
    
    except Exception as e:
        error_message = CustomException("Failed to craete new vectorstore " , e)
        logger.error(str(error_message))
    
def load_vectorstore():
    try:
        logger.info("Loading vectorstore")
        embedding_model = get_embedding_model()
        db = FAISS.load_local(vector_store_path, embedding_model, allow_dangerous_deserialization=True)
        logger.info("Vectorstore loaded successfully")
        return db
    
    except Exception as e:
        logger.error(f"Error loading vectorstore: {str(e)}")
        raise CustomException(f"Error loading vectorstore: {str(e)}")
