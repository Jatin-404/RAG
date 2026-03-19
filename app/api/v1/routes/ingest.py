import uuid
from fastapi import APIRouter, UploadFile, File, Form, Depends
from sqlalchemy.orm import Session
import tempfile, os
from pathlib import Path

from app.api.v1.dependencies import get_db
from app.services.ingestor import extract_text, chunk_text
from app.services.embedder import embed_chunks
from app.services.vectorstore import save_chunks

router = APIRouter()

@router.get("/health")
def ingest_health():
    return {"service": "ingest", "status": "ok"}

@router.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    department: str = Form("general"),
    domain: str = Form("general"),
    db: Session = Depends(get_db)
):
    # Save uploaded file to temp location
    suffix = Path(file.filename).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:   #FastAPI receives the file. But you can't process a file that's floating in memory — Unstructured needs a real file path on disk. So you save it to a temporary file first.
        tmp.write(await file.read())
        tmp_path = tmp.name

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

        return {
            "status": "success",
            "filename": file.filename,
            "chunks_stored": count
        }
    finally:
        os.unlink(tmp_path)  # always delete temp file