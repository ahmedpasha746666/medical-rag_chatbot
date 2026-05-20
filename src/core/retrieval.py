from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from src.core.llm import llm_model
from src.core.vectore_store import load_vectorstore
from src.common.logger import get_logger
from src.common.custom_exception import CustomException

logger = get_logger(__name__)

# ── Improved Prompt ───────────────────────────────────────────────────────────
PROMPT_TEMPLATE = """
You are an expert medical AI assistant. A doctor or patient is asking you a question
based on uploaded medical documents.

Instructions:
- Answer ONLY from the provided context.
- Be clear, structured, and medically accurate.
- Use bullet points or numbered steps when listing information.
- If the answer is not in the context, say:
  "I could not find this information in the uploaded documents."
- Do NOT make up information.

Context:
{context}

Question:
{question}

Answer:
"""


def chat_prompt_template():
    return PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    )


def qa_chain():
    try:
        logger.info("Initializing QA chain")
        model = llm_model()
        db    = load_vectorstore()

        chain = RetrievalQA.from_chain_type(
            llm=model,
            chain_type="stuff",
            retriever=db.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True,
            chain_type_kwargs={"prompt": chat_prompt_template()},
        )

        logger.info("QA chain ready")
        return chain

    except Exception as e:
        logger.error(f"Error creating QA chain: {str(e)}")
        raise CustomException(f"Error creating QA chain: {str(e)}")