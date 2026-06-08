import logging
import uuid
from typing import List, Dict, Any, Optional

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    SparseVectorParams, SparseIndexParams,
    NamedVector, NamedSparseVector,
    SparseVector, Filter, FieldCondition,
    MatchValue, SearchRequest, SearchParams,
    HnswConfigDiff,
)

from app.core.config import settings
from app.services.embeddings import get_embeddings, get_query_embedding

logger = logging.getLogger(__name__)

DENSE_NAME = "dense"
SPARSE_NAME = "sparse"


# def _build_sparse_vector(text: str) -> SparseVector:
#     """
#     Simple BM25-style sparse vector using term frequency.
#     In production replace with a real sparse encoder (e.g. SPLADE or BM25).
#     """
#     import re
#     from collections import Counter
#     tokens = re.findall(r"\w+", text.lower())
#     counts = Counter(tokens)
#     # Map token string to deterministic int index via hash
#     indices, values = [], []
#     for token, freq in counts.items():
#         idx = abs(hash(token)) % 50_000
#         indices.append(idx)
#         values.append(float(freq))
#     return SparseVector(indices=indices, values=values)

from collections import defaultdict

def _build_sparse_vector(text: str):
    import re

    sparse = defaultdict(float)

    for token in re.findall(r"\w+", text.lower()):
        idx = abs(hash(token)) % 50000
        sparse[idx] += 1.0

    indices = sorted(sparse.keys())
    values = [sparse[i] for i in indices]

    return SparseVector(indices=indices, values=values)
    
class VectorStore:
    def __init__(self):
        self.client = AsyncQdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
        )
        self.collection = settings.collection_name

    async def ensure_collection(self):
        existing = [c.name for c in (await self.client.get_collections()).collections]
        if self.collection not in existing:
            await self.client.create_collection(
                collection_name=self.collection,
                vectors_config={
                    DENSE_NAME: VectorParams(
                        size=settings.vector_size,
                        distance=Distance.COSINE,
                        hnsw_config=HnswConfigDiff(m=16, ef_construct=100),
                    )
                },
                sparse_vectors_config={
                    SPARSE_NAME: SparseVectorParams(
                        index=SparseIndexParams(on_disk=False)
                    )
                },
            )
            logger.info(f"Created collection: {self.collection}")

    async def upsert_records(self, records: List[Dict[str, Any]]):
        """Embed and upsert a list of {text, metadata} records."""
        texts = [r["text"] for r in records]
        dense_vecs = await get_embeddings(texts)

        points = []
        for i, record in enumerate(records):
            sparse_vec = _build_sparse_vector(record["text"])
            points.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector={
                        DENSE_NAME: dense_vecs[i],
                        SPARSE_NAME: sparse_vec,
                    },
                    payload={**record["metadata"], "text": record["text"]},
                )
            )

        # Batch upsert in chunks of 100
        batch = 100
        for i in range(0, len(points), batch):
            await self.client.upsert(
                collection_name=self.collection,
                points=points[i: i + batch],
            )
        logger.info(f"Upserted {len(points)} vectors")

    async def search(
        self,
        query: str,
        top_k: int = 5,
        mode: str = "hybrid",
        filters: Optional[Dict] = None,
    ) -> List[Dict]:
        """
        mode: dense | sparse | hybrid
        Returns list of {text, score, metadata} dicts.
        """
        qdrant_filter = None
        if filters:
            conditions = [
                FieldCondition(key=k, match=MatchValue(value=v))
                for k, v in filters.items()
            ]
            qdrant_filter = Filter(must=conditions)

        if mode == "dense":
            return await self._dense_search(query, top_k, qdrant_filter)
        elif mode == "sparse":
            return await self._sparse_search(query, top_k, qdrant_filter)
        else:
            return await self._hybrid_search(query, top_k, qdrant_filter)

    async def _dense_search(self, query: str, top_k: int, filt) -> List[Dict]:
        q_vec = await get_query_embedding(query)
        results = await self.client.search(
            collection_name=self.collection,
            query_vector=NamedVector(name=DENSE_NAME, vector=q_vec),
            limit=top_k,
            query_filter=filt,
            with_payload=True,
        )
        return [_hit_to_dict(r) for r in results]

    async def _sparse_search(self, query: str, top_k: int, filt) -> List[Dict]:
        sparse_vec = _build_sparse_vector(query)
        results = await self.client.search(
            collection_name=self.collection,
            query_vector=NamedSparseVector(name=SPARSE_NAME, vector=sparse_vec),
            limit=top_k,
            query_filter=filt,
            with_payload=True,
        )
        return [_hit_to_dict(r) for r in results]

    async def _hybrid_search(self, query: str, top_k: int, filt) -> List[Dict]:
        """Reciprocal rank fusion of dense + sparse results."""
        dense_results = await self._dense_search(query, top_k * 2, filt)
        sparse_results = await self._sparse_search(query, top_k * 2, filt)

        # RRF scoring
        k = 60
        scores: Dict[str, float] = {}
        payloads: Dict[str, Dict] = {}

        for rank, hit in enumerate(dense_results):
            uid = _uid(hit)
            scores[uid] = scores.get(uid, 0) + 1 / (k + rank + 1)
            payloads[uid] = hit

        for rank, hit in enumerate(sparse_results):
            uid = _uid(hit)
            scores[uid] = scores.get(uid, 0) + 1 / (k + rank + 1)
            payloads[uid] = hit

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        results = []
        for uid, rrf_score in ranked:
            item = dict(payloads[uid])
            item["score"] = rrf_score
            results.append(item)
        return results

    async def collection_info(self) -> Dict:
        info = await self.client.get_collection(self.collection)
        return {
            "total_vectors": info.vectors_count or 0,
            "status": str(info.status),
        }

    async def delete_by_pdf_id(self, pdf_id: str):
        from qdrant_client.models import FilterSelector
        await self.client.delete(
            collection_name=self.collection,
            points_selector=FilterSelector(
                filter=Filter(
                    must=[FieldCondition(key="pdf_id", match=MatchValue(value=pdf_id))]
                )
            ),
        )


def _hit_to_dict(hit) -> Dict:
    p = hit.payload or {}
    return {
        "text": p.get("text", ""),
        "score": hit.score,
        "pdf_name": p.get("pdf_name", ""),
        "page_number": p.get("page_number", 0),
        "pdf_id": p.get("pdf_id", ""),
        "section_type": p.get("section_type", "text"),
        "language": p.get("language", "en"),
        "has_image": p.get("has_image", False),
    }


def _uid(hit: Dict) -> str:
    return f"{hit['pdf_id']}_{hit['page_number']}_{hit['text'][:30]}"
