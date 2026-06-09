from pydantic_settings import BaseSettings
from typing import Literal


class Settings(BaseSettings):
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    collection_name: str = "realestate_docs"

    llm_provider: Literal["gemini", "openai", "anthropic"] = "gemini"
    embedding_provider: Literal["openai", "gemini"] = "gemini"

    openai_api_key: str = ""
    anthropic_api_key: str = ""
    gemini_api_key: str = ""

    upload_dir: str = "/app/uploads"
    chunk_size: int = 600
    chunk_overlap: int = 80

    # 3072 for gemini-embedding-001, 1536 for OpenAI, 768 for text-embedding-004
    vector_size: int = 3072

    redis_url: str = "redis://redis:6379/0"

    class Config:
        env_file = ".env"


settings = Settings()