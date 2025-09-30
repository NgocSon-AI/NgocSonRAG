from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
        env_file_encoding="utf-8"
    )

    # --- GROQ Configuration ---
    GROQ_API_KEY: str
    GROQ_MODEL: str = "llama-3.3-70b-versatile"

    # --- OPENAI Configuration ---
    OPENAI_API_KEY: str
    EMBEDDING_MODEL: str = "text-embedding-3-small"

    # --- RAG / Vector Store Configuration ---
    VECTOR_DIM: int = 1536
    VECTOR_INDEX_PATH: str = "data/faiss.index"
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 100
    TOP_K: int = 5

@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Get application settings
    """
    return Settings()
