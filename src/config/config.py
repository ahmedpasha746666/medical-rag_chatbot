from dotenv import load_dotenv
import os

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
hf_token = os.getenv("HF_TOKEN")
huggingface_hub_token = os.getenv("HUGGINGFACE_HUB_TOKEN")


data_path = r"C:\Users\Ahmed Pasha\Desktop\medical_rag_project\data/"
vector_store_path = r"C:\Users\Ahmed Pasha\Desktop\medical_rag_project\vectore_store/"
