# 🏥 Medical RAG Chatbot

> An advanced Retrieval-Augmented Generation (RAG) system for medical document Q&A, featuring hybrid search, query rewriting, semantic caching, and a real-time Streamlit interface.

---

## 📌 Project Overview

The **Medical RAG Chatbot** is a production-grade AI assistant designed for healthcare professionals. It ingests medical PDFs (including scanned documents via OCR), generates dense semantic embeddings, and enables intelligent question-answering grounded in the uploaded documents — all through an interactive web interface.

---

## 🗂️ Project Structure

```
medical-rag-chatbot/
│
├── venv/                          # Python virtual environment
│
├── data/
│   ├── uploads/                   # Raw uploaded PDF files
│   ├── processed/                 # Cleaned & chunked text data
│   └── chat_history/              # Persisted conversation logs
│
├── vector_store/                  # FAISS index & metadata storage
│
├── src/
│   ├── common/
│   │   ├── logger.py              # Centralized logging setup
│   │   ├── custom_exception.py    # Custom error classes
│   │   └── helpers.py            # Shared utility functions
│   │
│   ├── config/
│   │   └── config.py             # App configuration & env variables
│   │
│   ├── core/
│   │   ├── pdf_loader.py          # PDF ingestion (incl. scanned OCR)
│   │   ├── chunking.py            # Semantic text chunking strategies
│   │   ├── embeddings.py          # HuggingFace embedding generation
│   │   ├── vector_store.py        # FAISS index management
│   │   ├── retrieval.py           # Vector similarity search
│   │   ├── llm.py                 # LLM integration (Groq / OpenAI)
│   │   ├── rag_pipeline.py        # End-to-end RAG orchestration
│   │   └── load_and_process.py    # Document ingestion pipeline
│   │
│   ├── advanced_rag/
│   │   ├── hybrid_search.py       # Combines dense + sparse retrieval
│   │   ├── query_rewriter.py      # Query expansion & reformulation
│   │   ├── reranker.py            # Cross-encoder result reranking
│   │   ├── semantic_cache.py      # Cache similar query responses
│   │   ├── memory.py              # Multi-turn conversation memory
│   │   ├── guardrails.py          # Input/output safety filters
│   │   └── streaming.py           # Token streaming support
│   │
│   ├── frontend/
│   │   └── streamlit_app.py       # Interactive web UI
│   │
│   └── utils/
│       └── file_handler.py        # File upload & management helpers
│
├── .env                           # API keys & environment config
├── requirements.txt               # Python dependencies
├── README.md                      # Project documentation
└── setup.py                       # Package setup
```

---

## ✨ Key Features

| Feature | Description |
|---|---|
| 📄 **PDF Ingestion** | Upload single or multiple PDFs; scanned docs handled via `pytesseract` OCR |
| 🧩 **Semantic Chunking** | Intelligent text splitting preserving medical context |
| 🔢 **Dense Embeddings** | HuggingFace `sentence-transformers` for high-quality vector representations |
| ⚡ **FAISS Vector Store** | Fast approximate nearest-neighbor search for large document sets |
| 🔍 **Hybrid Search** | Combines dense semantic search + sparse BM25 keyword matching |
| ✏️ **Query Rewriting** | Reformulates queries for improved retrieval accuracy |
| 🏆 **Reranking** | Cross-encoder reranking of retrieved chunks for precision |
| 🗄️ **Semantic Cache** | Caches responses for similar queries, reducing LLM API costs |
| 💬 **Conversation Memory** | Multi-turn chat with context-aware follow-up handling |
| 🛡️ **Guardrails** | Input/output safety filters for medical domain compliance |
| 🌊 **Streaming** | Real-time token-by-token response streaming |
| 🖥️ **Streamlit UI** | Clean, interactive web interface for healthcare professionals |

---

## 🛠️ Tech Stack

- **Language:** Python 3.10+
- **LLMs:** Groq (Llama / Mixtral), OpenAI API, Claude API
- **Embeddings:** HuggingFace `sentence-transformers`
- **Vector DB:** FAISS
- **RAG Framework:** LangChain
- **OCR:** `pytesseract` + `pdf2image`
- **Frontend:** Streamlit
- **Backend API:** FastAPI
- **Environment:** python-dotenv

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/medical-rag-chatbot.git
cd medical-rag-chatbot
```

### 2. Create & Activate Virtual Environment

```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the root directory:

```env
GROQ_API_KEY=your_groq_api_key_here
OPENAI_API_KEY=your_openai_api_key_here   # optional
HUGGINGFACE_TOKEN=your_hf_token_here      # optional for private models
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
FAISS_INDEX_PATH=vector_store/
CHUNK_SIZE=512
CHUNK_OVERLAP=64
```

### 5. Run the Application

```bash
streamlit run src/frontend/streamlit_app.py
```

Open your browser at `http://localhost:8501`

---

## 🧠 Architecture

```
User Upload (PDF)
      │
      ▼
PDF Loader + OCR (pytesseract)
      │
      ▼
Semantic Chunking
      │
      ▼
HuggingFace Embeddings
      │
      ▼
FAISS Vector Store
      │
User Query ──► Query Rewriter ──► Hybrid Search (Dense + BM25)
                                        │
                                        ▼
                                   Reranker
                                        │
                                        ▼
                              Context + Memory + Guardrails
                                        │
                                        ▼
                                 LLM (Groq/OpenAI)
                                        │
                                        ▼
                              Streaming Response ──► Streamlit UI
                                        │
                                        ▼
                               Semantic Cache (store)
```

---

## 📦 Requirements

```
langchain
langchain-groq
langchain-openai
langchain-community
faiss-cpu
sentence-transformers
huggingface-hub
streamlit
fastapi
uvicorn
pytesseract
pdf2image
Pillow
python-dotenv
rank-bm25
```

---

## 🔮 Advanced RAG Features Explained

### Hybrid Search (`hybrid_search.py`)
Combines FAISS dense vector retrieval with BM25 sparse keyword matching. Results are merged using Reciprocal Rank Fusion (RRF) for better coverage of both semantic and lexical matches.

### Query Rewriter (`query_rewriter.py`)
Uses an LLM to expand and reformulate user queries before retrieval, improving recall for medical terminology and abbreviations.

### Reranker (`reranker.py`)
A cross-encoder model (e.g., `cross-encoder/ms-marco-MiniLM-L-6-v2`) re-scores retrieved chunks against the query for precision before passing context to the LLM.

### Semantic Cache (`semantic_cache.py`)
Embeds incoming queries and checks cosine similarity against cached queries. If similarity exceeds a threshold, returns the cached answer instantly — saving LLM API costs for repeated questions.

### Guardrails (`guardrails.py`)
Validates that inputs are medically relevant and outputs don't contain harmful or hallucinatory information outside document context.

---

## 👤 Author

**Ahmed Pasha** — AI/ML Engineer  
📧 ahmedpasha99999@gmail.com | 📱 +91-9535109746 | 📍 Hyderabad, India

---

## 📄 License

This project is for educational and professional portfolio purposes.
