import os
import uuid
import shutil
import logging
from fastapi import APIRouter, UploadFile, File, HTTPException
from celery.result import AsyncResult

from app.core.config import settings
from app.worker.tasks import ingest_pdf_task
from app.worker.celery_app import celery_app
from app.services.vector_store import VectorStore
from app.models.schemas import IngestResponse, IngestAccepted

router = APIRouter(prefix="/ingest", tags=["Ingest"])
logger = logging.getLogger(__name__)
store = VectorStore()


@router.post("/upload", response_model=IngestAccepted, status_code=202)
async def upload_pdf(file: UploadFile = File(...)):
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

    task = ingest_pdf_task.delay(pdf_id, file.filename, save_path)

    return IngestAccepted(
        task_id=task.id,
        pdf_id=pdf_id,
        pdf_name=file.filename,
        status="queued",
        message="PDF queued for processing. Poll /ingest/status/{task_id} for updates.",
    )


@router.get("/status/{task_id}")
async def task_status(task_id: str):
    result = AsyncResult(task_id, app=celery_app)

    if result.state == "PENDING":
        return {"task_id": task_id, "status": "pending", "step": "Waiting in queue..."}
    elif result.state == "PROCESSING":
        meta = result.info or {}
        return {
            "task_id": task_id,
            "status": "processing",
            "step": meta.get("step", "Processing..."),
            "total_chunks": meta.get("total_chunks"),
        }
    elif result.state == "SUCCESS":
        return {"task_id": task_id, **result.result}
    elif result.state == "FAILURE":
        return {"task_id": task_id, "status": "failed", "error": str(result.result)}

    return {"task_id": task_id, "status": result.state}


@router.delete("/{pdf_id}")
async def delete_pdf(pdf_id: str):
    """Remove all vectors for a specific PDF from the vector store."""
    await store.delete_by_pdf_id(pdf_id)
    pdf_path = os.path.join(settings.upload_dir, f"{pdf_id}.pdf")
    if os.path.exists(pdf_path):
        os.remove(pdf_path)
    return {"status": "deleted", "pdf_id": pdf_id}
