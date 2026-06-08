from pydantic import BaseModel
from typing import Optional, List


class ChunkMetadata(BaseModel):
    pdf_id: str
    pdf_name: str
    page_number: int
    chunk_index: int
    total_pages: int
    language: str
    has_image: bool
    section_type: str          # text | table | diagram | mixed
    char_count: int
    source_path: str


class IngestResponse(BaseModel):
    pdf_id: str
    pdf_name: str
    total_pages: int
    total_chunks: int
    languages_detected: List[str]
    pages_with_images: int
    status: str
    warnings: List[str] = []


class QueryRequest(BaseModel):
    query: str
    top_k: int = 5
    search_mode: str = "hybrid"   # dense | sparse | hybrid
    llm_provider: Optional[str] = None   # override per-request


class SearchResult(BaseModel):
    chunk_text: str
    score: float
    pdf_name: str
    page_number: int
    pdf_id: str
    section_type: str
    language: str
    has_image: bool


class QueryResponse(BaseModel):
    answer: str
    sources: List[SearchResult]
    query: str
    search_mode: str
    llm_used: str


class HealthResponse(BaseModel):
    status: str
    qdrant: str
    collection: str
    total_vectors: int
