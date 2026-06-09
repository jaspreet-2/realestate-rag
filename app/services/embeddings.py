import logging
from typing import List
from app.core.config import settings

logger = logging.getLogger(__name__)

_EMBED_PRICE_PER_1K_CHARS = 0.000025   # USD per 1,000 characters (text-embedding-004, Vertex AI)



async def get_embeddings(texts: List[str]) -> List[List[float]]:
    """Return dense embeddings for a list of texts."""
    provider = settings.embedding_provider

    if provider == "openai":
        return await _openai_embeddings(texts)
    elif provider == "gemini":
        return await _gemini_embeddings(texts)
    else:
        raise ValueError(f"Unknown embedding provider: {provider}")


async def _openai_embeddings(texts: List[str]) -> List[List[float]]:
    from openai import AsyncOpenAI
    client = AsyncOpenAI(api_key=settings.openai_api_key)

    # OpenAI allows up to 2048 inputs per call; batch in 100s for safety
    all_embeddings = []
    batch_size = 100
    for i in range(0, len(texts), batch_size):
        batch = texts[i: i + batch_size]
        response = await client.embeddings.create(
            model="text-embedding-3-small",
            input=batch,
        )
        all_embeddings.extend([item.embedding for item in response.data])
    return all_embeddings


def _get_gemini_client():
    import os
    from google import genai

    if settings.google_application_credentials:
        os.environ.setdefault(
            "GOOGLE_APPLICATION_CREDENTIALS", settings.google_application_credentials
        )
    return genai.Client(
        vertexai=True,
        project=settings.google_cloud_project,
        location=settings.google_cloud_location,
    )


async def _gemini_embeddings(texts: List[str]) -> List[List[float]]:
    from google.genai import types

    client = _get_gemini_client()
    all_embeddings = []
    total_chars = 0
    for text in texts:
        result = client.models.embed_content(
            model=settings.gemini_embedding_model,
            contents=text,
            config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
        )
        all_embeddings.append(list(result.embeddings[0].values))
        total_chars += len(text)

    cost = (total_chars / 1_000) * _EMBED_PRICE_PER_1K_CHARS
    logger.info(
        "Embedding cost [%s] — %d chunks | %d chars | ~$%.6f",
        settings.gemini_embedding_model, len(texts), total_chars, cost,
    )
    return all_embeddings


async def get_query_embedding(query: str) -> List[float]:
    """Single query embedding (task_type=retrieval_query for Gemini)."""
    provider = settings.embedding_provider

    if provider == "openai":
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=settings.openai_api_key)
        response = await client.embeddings.create(
            model="text-embedding-3-small",
            input=[query],
        )
        return response.data[0].embedding

    elif provider == "gemini":
        from google.genai import types

        client = _get_gemini_client()
        result = client.models.embed_content(
            model=settings.gemini_embedding_model,
            contents=query,
            config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
        )
        cost = (len(query) / 1_000) * _EMBED_PRICE_PER_1K_CHARS
        logger.info(
            "Query embedding cost [%s] — %d chars | ~$%.6f",
            settings.gemini_embedding_model, len(query), cost,
        )
        return list(result.embeddings[0].values)

    raise ValueError(f"Unknown embedding provider: {provider}")
