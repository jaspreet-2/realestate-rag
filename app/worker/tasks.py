import asyncio
import logging
import os

from app.worker.celery_app import celery_app
from app.services.pdf_processor import process_pdf
from app.services.vector_store import VectorStore
from app.core.config import settings

logger = logging.getLogger(__name__)


def _run_async(coro):
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            raise RuntimeError
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)


@celery_app.task(
    bind=True,
    name="ingest_pdf",
    max_retries=3,
    default_retry_delay=10,
)
def ingest_pdf_task(self, pdf_id: str, pdf_name: str, save_path: str):
    try:
        self.update_state(state="PROCESSING", meta={"step": "Extracting text from PDF..."})

        result = process_pdf(
            file_path=save_path,
            pdf_id=pdf_id,
            pdf_name=pdf_name,
            chunk_size=settings.chunk_size,
            overlap=settings.chunk_overlap,
        )

        if not result["valid"]:
            if os.path.exists(save_path):
                os.remove(save_path)
            return {"status": "failed", "pdf_id": pdf_id, "reason": result["reason"]}

        if not result["records"]:
            if os.path.exists(save_path):
                os.remove(save_path)
            return {"status": "failed", "pdf_id": pdf_id, "reason": "No readable text found."}

        self.update_state(
            state="PROCESSING",
            meta={
                "step": f"Embedding {len(result['records'])} chunks...",
                "total_chunks": len(result["records"]),
            },
        )

        store = VectorStore()
        _run_async(store.ensure_collection())
        _run_async(store.upsert_records(result["records"]))

        return {
            "status": "success",
            "pdf_id": pdf_id,
            "pdf_name": pdf_name,
            "total_pages": result["total_pages"],
            "total_chunks": len(result["records"]),
            "languages_detected": result["languages"],
            "pages_with_images": result["pages_with_images"],
            "warnings": result["warnings"],
        }

    except Exception as exc:
        logger.error(f"Task failed for {pdf_id}: {exc}")
        raise self.retry(exc=exc)