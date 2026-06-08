# Real Estate Document Intelligence API

FastAPI + Qdrant + Gemini/OpenAI/Anthropic  
Supports: English, Hindi, Bengali, Marathi, and any Tesseract-supported language  
Features: blur detection, contrast enhancement, OCR fallback, dense + sparse + hybrid search

---

## Project Structure

```
realestate-rag/
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── .env.example
└── app/
    ├── main.py
    ├── core/config.py
    ├── models/schemas.py
    ├── services/
    │   ├── pdf_processor.py   ← blur detect, OCR, chunking, validation
    │   ├── embeddings.py      ← OpenAI / Gemini embeddings
    │   ├── vector_store.py    ← Qdrant dense+sparse+hybrid search
    │   └── llm.py             ← Gemini / OpenAI / Anthropic answer gen
    └── api/
        ├── ingest.py          ← POST /ingest/upload
        ├── query.py           ← POST /query/
        └── health.py          ← GET /health
```

---

## Quick Start

### 1. Configure environment

```bash
cp .env.example .env
# Edit .env — fill in at least one LLM key and one embedding key
```

Key settings:
| Variable | Values | Default |
|---|---|---|
| `LLM_PROVIDER` | `gemini` / `openai` / `anthropic` | `gemini` |
| `EMBEDDING_PROVIDER` | `openai` / `gemini` | `openai` |
| `OPENAI_API_KEY` | your key | — |
| `GEMINI_API_KEY` | your key | — |
| `ANTHROPIC_API_KEY` | your key | — |

> If using **Gemini embeddings**, set `EMBEDDING_PROVIDER=gemini` and `vector_size=768` in config.py (Gemini gemini-embedding-001 is 768-dim).  
> If using **OpenAI embeddings** (default), keep `vector_size=1536`.

### 2. Start services

```bash
docker-compose up --build
```

API runs at: http://localhost:8000  
Swagger UI: http://localhost:8000/docs  
Qdrant dashboard: http://localhost:6333/dashboard

---

## API Reference

### Upload a PDF
```bash
curl -X POST http://localhost:8000/ingest/upload \
  -F "file=@property_deed.pdf"
```

Response:
```json
{
  "pdf_id": "abc-123",
  "pdf_name": "property_deed.pdf",
  "total_pages": 12,
  "total_chunks": 47,
  "languages_detected": ["en", "hi"],
  "pages_with_images": 3,
  "status": "success",
  "warnings": ["Page 4 was blurry — contrast enhanced before OCR."]
}
```

### Query documents
```bash
curl -X POST http://localhost:8000/query/ \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the total area of the plot in survey number 42?",
    "top_k": 5,
    "search_mode": "hybrid",
    "llm_provider": "gemini"
  }'
```

`search_mode` options:
- `dense` — semantic similarity only (OpenAI/Gemini embeddings)
- `sparse` — keyword/BM25-style matching
- `hybrid` — Reciprocal Rank Fusion of both (recommended)

`llm_provider` is optional — overrides the default for this request.

Response:
```json
{
  "answer": "The plot at survey number 42 has a total area of 2,400 sq ft...",
  "sources": [
    {
      "chunk_text": "Survey No. 42, total area 2400 sq ft...",
      "score": 0.921,
      "pdf_name": "property_deed.pdf",
      "page_number": 3,
      "pdf_id": "abc-123",
      "section_type": "text",
      "language": "en",
      "has_image": false
    }
  ],
  "query": "...",
  "search_mode": "hybrid",
  "llm_used": "gemini"
}
```

### Raw search (no LLM)
```bash
GET http://localhost:8000/query/search-raw?q=plot+area&top_k=5&mode=hybrid
```

### Delete a PDF's vectors
```bash
curl -X DELETE http://localhost:8000/ingest/{pdf_id}
```

### Health check
```bash
GET http://localhost:8000/health
```

---

## PDF Quality Handling

| Condition | What happens |
|---|---|
| Clear native text | Extracted directly (fastest) |
| Blurry scan | Contrast × 2.5, sharpened, Otsu threshold, then OCR |
| No selectable text | Rendered to image, OCR via Tesseract |
| Hindi/Bengali | Tesseract uses `hin+eng` / `ben+eng` language pack |
| Non real-estate doc | Returns HTTP 422 with explanation message |
| Empty PDF | Returns HTTP 422 |
| Diagrams/images | Flagged in metadata (`has_image: true`, `section_type: diagram`) |

---

## Switching LLM/Embedding Provider

Per-request (query endpoint):
```json
{ "query": "...", "llm_provider": "openai" }
```

Globally (in .env):
```
LLM_PROVIDER=anthropic
EMBEDDING_PROVIDER=openai
```

> **Note**: Changing `EMBEDDING_PROVIDER` after data is indexed requires re-ingesting all PDFs, since vector dimensions differ between providers.

---

## Production Tips

- Replace `_build_sparse_vector()` in `vector_store.py` with a real SPLADE or BM25 model for better sparse search quality.
- Add authentication (API key header) in `app/main.py`.
- Set `QDRANT_HOST` to a managed Qdrant Cloud instance for production.
- Add `celery` + `redis` for async background ingestion of large PDFs.
