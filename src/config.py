import os
from pathlib import Path

# Base Project Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DOCUMENTS_DIR = DATA_DIR / "documents"
VECTOR_STORE_PATH = DATA_DIR / "vector_store.json"

# Ingestion & Chunking Defaults
DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 80
SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf", ".py", ".json", ".csv"}

# Embedding Model Configuration (Dense Search)
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MULTILINGUAL_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Hybrid Retrieval Settings (Dense + Sparse BM25 + Reciprocal Rank Fusion)
HYBRID_SEARCH_ENABLED = True
RRF_K = 60  # Standard constant for Reciprocal Rank Fusion
DENSE_WEIGHT = 0.65
SPARSE_WEIGHT = 0.35

# Re-ranking Configuration (Cross-Encoder / Precision Layer)
RERANKING_ENABLED = True
INITIAL_RETRIEVAL_K = 8
FINAL_TOP_K = 4
DEFAULT_SIMILARITY_THRESHOLD = 0.18

# RAG Triad & Quality Evaluation Thresholds
EVALUATION_ENABLED = True
MIN_GROUNDEDNESS_SCORE = 0.70
MIN_CONTEXT_RELEVANCE = 0.25

# Local LLM Inference Configuration
DEFAULT_LOCAL_ENDPOINT = "http://localhost:5272/v1/chat/completions"
DEFAULT_MODEL_NAME = "phi-3-mini"
DEFAULT_TEMPERATURE = 0.2
DEFAULT_MAX_TOKENS = 800

# Ensure directories exist
os.makedirs(DOCUMENTS_DIR, exist_ok=True)
