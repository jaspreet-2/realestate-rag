import logging
from fastapi import APIRouter, HTTPException
from app.services.vector_store import VectorStore
from app.services.llm import generate_answer
from app.models.schemas import QueryRequest, QueryResponse, SearchResult

router = APIRouter(prefix="/query", tags=["Query"])
logger = logging.getLogger(__name__)
store = VectorStore()


@router.post("/", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """
    Query your uploaded documents.
    - search_mode: 'dense' | 'sparse' | 'hybrid'
    - llm_provider: 'gemini' | 'openai' | 'anthropic' (optional, overrides default)
    - Returns AI-generated answer + source chunks with page numbers
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    valid_modes = {"dense", "sparse", "hybrid"}
    if request.search_mode not in valid_modes:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid search_mode. Choose from: {valid_modes}"
        )

    try:
        chunks = await store.search(
            query=request.query,
            top_k=request.top_k,
            mode=request.search_mode,
        )
    except Exception as e:
        logger.error(f"Vector search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Search error: {e}")

    if not chunks:
        return QueryResponse(
            answer="No relevant documents found. Please upload real estate PDFs first.",
            sources=[],
            query=request.query,
            search_mode=request.search_mode,
            llm_used="none",
        )

    try:
        answer, llm_used = await generate_answer(
            query=request.query,
            chunks=chunks,
            provider=request.llm_provider,
        )
    except Exception as e:
        logger.error(f"LLM generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"LLM error: {e}")

    sources = [
        SearchResult(
            chunk_text=c["text"],
            score=round(c["score"], 4),
            pdf_name=c["pdf_name"],
            page_number=c["page_number"],
            pdf_id=c["pdf_id"],
            section_type=c["section_type"],
            language=c["language"],
            has_image=c["has_image"],
        )
        for c in chunks
    ]

    return QueryResponse(
        answer=answer,
        sources=sources,
        query=request.query,
        search_mode=request.search_mode,
        llm_used=llm_used,
    )


@router.get("/search-raw")
async def raw_search(q: str, top_k: int = 5, mode: str = "hybrid"):
    """Raw vector search — returns chunks without LLM synthesis."""
    chunks = await store.search(query=q, top_k=top_k, mode=mode)
    return {"results": chunks, "count": len(chunks)}
