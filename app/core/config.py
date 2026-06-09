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

    # Google Cloud credentials (use instead of gemini_api_key)
    # Set GOOGLE_APPLICATION_CREDENTIALS env var to the path of your service account JSON,
    # or leave empty to use gcloud ADC / GCE metadata server.
    google_application_credentials: str = ""
    google_cloud_project: str = ""
    google_cloud_location: str = "us-central1"

    # Gemini model names (configurable per deployment)
    gemini_model: str = "gemini-2.5-flash"
    gemini_embedding_model: str = "text-embedding-004"

    upload_dir: str = "/app/uploads"
    chunk_size: int = 600
    chunk_overlap: int = 80

    # 768 for text-embedding-004 (Vertex AI), 1536 for OpenAI
    vector_size: int = 768

    redis_url: str = "redis://redis:6379/0"

    class Config:
        env_file = ".env"


settings = Settings()