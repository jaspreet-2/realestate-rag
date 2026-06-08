import os
import uuid
import shutil
import logging
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from app.core.config import settings
from app.services.pdf_processor import process_pdf
from app.services.vector_store import VectorStore
from app.models.schemas import IngestResponse

router = APIRouter(prefix="/ingest", tags=["Ingest"])
logger = logging.getLogger(__name__)
store = VectorStore()


@router.post("/upload", response_model=IngestResponse)
async def upload_pdf(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    """
    Upload a PDF (any language). The system will:
    1. Detect blur and enhance if needed
    2. OCR pages that lack selectable text
    3. Validate it's a real-estate document
    4. Chunk, embed, and store in Qdrant with full metadata
    """
    print(f"Received file: {file.filename}, content type: {file.content_type}")
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    os.makedirs(settings.upload_dir, exist_ok=True)
    pdf_id = str(uuid.uuid4())
    save_path = os.path.join(settings.upload_dir, f"{pdf_id}.pdf")

    try:
        with open(save_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {e}")

    result = process_pdf(
        file_path=save_path,
        pdf_id=pdf_id,
        pdf_name=file.filename,
        chunk_size=settings.chunk_size,
        overlap=settings.chunk_overlap,
    )

    if not result["valid"]:
        os.remove(save_path)
        raise HTTPException(status_code=422, detail=result["reason"])

    if not result["records"]:
        os.remove(save_path)
        raise HTTPException(status_code=422, detail="No readable text found in this PDF.")

    await store.ensure_collection()
    await store.upsert_records(result["records"])

    return IngestResponse(
        pdf_id=pdf_id,
        pdf_name=file.filename,
        total_pages=result["total_pages"],
        total_chunks=len(result["records"]),
        languages_detected=result["languages"],
        pages_with_images=result["pages_with_images"],
        status="success",
        warnings=result["warnings"],
    )


@router.delete("/{pdf_id}")
async def delete_pdf(pdf_id: str):
    """Remove all vectors for a specific PDF from the vector store."""
    await store.delete_by_pdf_id(pdf_id)
    pdf_path = os.path.join(settings.upload_dir, f"{pdf_id}.pdf")
    if os.path.exists(pdf_path):
        os.remove(pdf_path)
    return {"status": "deleted", "pdf_id": pdf_id}
