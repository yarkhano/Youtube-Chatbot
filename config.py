from dotenv import load_dotenv

load_dotenv()

MODEL_NAME = "gemini-2.5-flash"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 300
RETRIEVAL_K = 4
HISTORY_MESSAGES = 12