import uuid
import tempfile
import os
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, Form, Depends
from sqlalchemy.orm import Session

from app.api.v1.dependencies import get_db
from app.services.ingestor import extract_text, chunk_text
from app.services.embedder import embed_chunks
from app.services.vectorstore import save_chunks
from app.tasks.ingest_tasks import ingest_file_task

router = APIRouter()

@router.get("/health")
def ingest_health():
    return {"service": "ingest", "status": "ok"}

@router.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    department: str = Form("general"),
    domain: str = Form("general"),
    background: bool = Form(False),
    db: Session = Depends(get_db)
):
    suffix = Path(file.filename).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    # background=True → Celery job, returns immediately
    if background:
        task = ingest_file_task.delay(tmp_path, file.filename, department, domain)
        return {
            "status": "queued",
            "job_id": task.id,
            "filename": file.filename
        }

    # background=False → synchronous, waits for completion (default)
    try:
        text = extract_text(tmp_path)
        chunks = chunk_text(text)
        embeddings = embed_chunks(chunks)
        metadata = {
            "document_id": str(uuid.uuid4()),
            "filename": file.filename,
            "department": department,
            "domain": domain,
            "custom_fields": {"department": department}
        }
        count = save_chunks(db, chunks, embeddings, metadata)
        return {"status": "success", "filename": file.filename, "chunks_stored": count}
    finally:
        os.unlink(tmp_path)

@router.get("/status/{job_id}")
def job_status(job_id: str):
    task = ingest_file_task.AsyncResult(job_id)
    return {
        "job_id": job_id,
        "status": task.status,
        "result": task.result if task.ready() else task.info
    }