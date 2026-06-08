import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import ingest, query, health
from app.core.config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

app = FastAPI(
    title="Real Estate Document Intelligence API",
    description=(
        "Upload multi-language property PDFs (Hindi, English, Bengali, etc.), "
        "store in Qdrant vector DB, and query with dense/sparse/hybrid search + "
        "Gemini / OpenAI / Anthropic LLM."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(ingest.router)
app.include_router(query.router)


@app.get("/")
async def root():
    return {
        "message": "Real Estate Document Intelligence API",
        "docs": "/docs",
        "llm_provider": settings.llm_provider,
        "embedding_provider": settings.embedding_provider,
    }
