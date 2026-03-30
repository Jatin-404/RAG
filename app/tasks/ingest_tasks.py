import uuid
import os
from pathlib import Path
from celery import Celery
from app.core.config import settings
from app.db.session import SessionLocal
from app.services.ingestor import extract_chunks, extract_json_metadata
from app.services.embedder import embed_chunks
from app.services.vectorstore import save_chunks
from app.services.classifier import classify_document

celery_app = Celery(
    "rag_tasks",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL
)

TABULAR_TYPES = {'.csv', '.xlsx', '.xls', '.ods'}

def build_classifier_text(file_path: str, filename: str, chunks: list[str]) -> str:
    """
    For tabular files — prepend column names so classifier has enough signal.
    For all other files — use first 3 chunks as before.
    """
    ext = Path(filename).suffix.lower()

    if ext in TABULAR_TYPES:
        try:
            import pandas as pd
            if ext == '.csv':
                df = pd.read_csv(file_path)
            else:
                df = pd.read_excel(file_path, sheet_name=0)
            columns = list(df.columns)
            return f"Spreadsheet columns: {', '.join(columns)}\n\n" + " ".join(chunks[:3])
        except Exception:
            pass

    return " ".join(chunks[:3])


@celery_app.task(bind=True)
def ingest_file_task(self, file_path: str, filename: str):
    try:
        self.update_state(state="PROGRESS", meta={"step": "extracting chunks"})
        chunks = extract_chunks(file_path)
        json_meta = extract_json_metadata(file_path)

        self.update_state(state="PROGRESS", meta={"step": "classifying"})
        text_for_classifier = build_classifier_text(file_path, filename, chunks)
        classification = classify_document(text_for_classifier)

        resolved_department = classification.get("department") or "general"
        resolved_domain = (
            json_meta.get("detected_domain")
            or classification.get("domain")
            or "general"
        )
        custom_fields = {
            "department": resolved_department,
            **{k: v for k, v in json_meta.items() if k != "detected_domain"},
            **classification.get("custom_fields", {})
        }

        self.update_state(state="PROGRESS", meta={"step": "embedding"})
        embeddings = embed_chunks(chunks)

        self.update_state(state="PROGRESS", meta={"step": "saving to database"})
        db = SessionLocal()
        try:
            metadata = {
                "document_id": str(uuid.uuid4()),
                "filename": filename,
                "department": resolved_department,
                "domain": resolved_domain,
                "custom_fields": custom_fields
            }
            count = save_chunks(db, chunks, embeddings, metadata)
        finally:
            db.close()

        try:
            os.unlink(file_path)
        except PermissionError:
            pass

        return {
            "status": "success",
            "filename": filename,
            "chunks_stored": count,
            "department": resolved_department,
            "domain": resolved_domain
        }

    except Exception as e:
        if self.request.retries >= self.max_retries:
            try:
                if os.path.exists(file_path):
                    os.unlink(file_path)
            except PermissionError:
                pass
        raise self.retry(exc=e, countdown=5, max_retries=3)