import logging
from typing import List
from app.core.config import settings

logger = logging.getLogger(__name__)


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


async def _gemini_embeddings(texts: List[str]) -> List[List[float]]:
    import google.generativeai as genai
    genai.configure(api_key=settings.gemini_api_key)

    all_embeddings = []
    for text in texts:
        result = genai.embed_content(
            model="models/gemini-embedding-001",
            content=text,
            task_type="retrieval_document",
        )
        all_embeddings.append(result["embedding"])
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
        import google.generativeai as genai
        genai.configure(api_key=settings.gemini_api_key)
        result = genai.embed_content(
            model="models/gemini-embedding-001",
            content=query,
            task_type="retrieval_query",
        )
        return result["embedding"]

    raise ValueError(f"Unknown embedding provider: {provider}")
