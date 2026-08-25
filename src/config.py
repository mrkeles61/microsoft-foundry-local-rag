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

# Embedding Model Configuration
# Light and fast embedding model running 100% locally
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MULTILINGUAL_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Retrieval Hyperparameters
DEFAULT_TOP_K = 4
DEFAULT_SIMILARITY_THRESHOLD = 0.20

# Local LLM Inference Configuration
# Supports Microsoft Foundry Local (5272), Ollama (11434), LM Studio (1234), or OpenAI endpoints
DEFAULT_LOCAL_ENDPOINT = "http://localhost:5272/v1/chat/completions"
DEFAULT_MODEL_NAME = "phi-3-mini"
DEFAULT_TEMPERATURE = 0.2
DEFAULT_MAX_TOKENS = 750

# Ensure essential directories exist
os.makedirs(DOCUMENTS_DIR, exist_ok=True)
