from fastapi import APIRouter, HTTPException
from app.services.vector_store import VectorStore
from app.models.schemas import HealthResponse
from app.core.config import settings

router = APIRouter(tags=["Health"])
store = VectorStore()


@router.get("/health", response_model=HealthResponse)
async def health():
    try:
        await store.ensure_collection()
        info = await store.collection_info()
        return HealthResponse(
            status="ok",
            qdrant="connected",
            collection=settings.collection_name,
            total_vectors=info["total_vectors"],
        )
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Qdrant unavailable: {e}")
