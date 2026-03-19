import uuid
import tempfile
import os
from celery import Celery
from app.core.config import settings
from app.db.session import SessionLocal
from app.services.ingestor import extract_text, chunk_text
from app.services.embedder import embed_chunks
from app.services.vectorstore import save_chunks

celery_app = Celery(
    "rag_tasks",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL
)

@celery_app.task(bind=True)
def ingest_file_task(self, file_path: str, filename: str, department: str, domain: str):
    try:
        self.update_state(state="PROGRESS", meta={"step": "extracting text"})
        text = extract_text(file_path)

        self.update_state(state="PROGRESS", meta={"step": "chunking"})
        chunks = chunk_text(text)

        self.update_state(state="PROGRESS", meta={"step": "embedding"})
        embeddings = embed_chunks(chunks)

        self.update_state(state="PROGRESS", meta={"step": "saving to database"})
        db = SessionLocal()
        try:
            metadata = {
                "document_id": str(uuid.uuid4()),
                "filename": filename,
                "department": department,
                "domain": domain,
                "custom_fields": {"department": department}
            }
            count = save_chunks(db, chunks, embeddings, metadata)
        finally:
            db.close()

        os.unlink(file_path)  # cleanup only on success

        return {
            "status": "success",
            "filename": filename,
            "chunks_stored": count
        }

    except Exception as e:
        if self.request.retries >= self.max_retries:
            # final failure — cleanup now
            if os.path.exists(file_path):
                os.unlink(file_path)
        raise self.retry(exc=e, countdown=5, max_retries=3)