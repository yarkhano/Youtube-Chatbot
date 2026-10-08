from pathlib import Path

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

MODEL_NAME = "gemini-2.5-flash"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 300
RETRIEVAL_K = 4

# Last three completed question-answer pairs.
HISTORY_MESSAGES = 6

CACHE_TTL = 3600
MAX_CACHED_VIDEOS = 10