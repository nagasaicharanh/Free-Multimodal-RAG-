"""
Configuration module for Free Multimodal RAG system.
Loads API keys and settings from environment variables.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# API Keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

# ChromaDB Configuration
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./data/chroma_db")
CHROMA_COLLECTION_NAME = "multimodal_rag"

# Embedding Model
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_MODEL_DEVICE = "cpu"

# LLM Configuration
GROQ_MODEL_NAME = "llama-3.3-70b-versatile"
GOOGLE_MODEL_NAME = "gemini-1.5-flash"

# Fallback LLM (Local)
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral")

# Processing Parameters
CHUNK_SIZE = 1024
CHUNK_OVERLAP = 200
MAX_TABLE_SUMMARY_LENGTH = 500
MAX_IMAGE_DESCRIPTION_LENGTH = 1000
TOP_K_RETRIEVAL = 3

# API Rate Limits
GROQ_RATE_LIMIT_TOKENS_PER_MINUTE = 6000
GEMINI_RATE_LIMIT_REQUESTS_PER_DAY = 1500
