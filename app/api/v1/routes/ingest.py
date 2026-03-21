import tempfile
from pathlib import Path
from fastapi import APIRouter, UploadFile, File

from app.tasks.ingest_tasks import ingest_file_task
from app.schemas.ingest import QueuedResponse, JobStatusResponse

router = APIRouter()

@router.get("/health")
def ingest_health():
    return {"service": "ingest", "status": "ok"}

@router.post("/upload", response_model=QueuedResponse)
async def upload_file(
    file: UploadFile = File(...)
):
    suffix = Path(file.filename).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    task = ingest_file_task.delay(tmp_path, file.filename)
    return QueuedResponse(status="queued", job_id=task.id, filename=file.filename)

@router.get("/status/{job_id}", response_model=JobStatusResponse)
def job_status(job_id: str):
    task = ingest_file_task.AsyncResult(job_id)

    def _json_safe(value):
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, Exception):
            return {"error_type": type(value).__name__, "error": str(value)}
        if isinstance(value, dict):
            return {str(k): _json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [_json_safe(v) for v in value]
        # Celery can return rich objects (including exceptions); stringify as last resort.
        return str(value)

    if task.status == "FAILURE":
        result = _json_safe(task.result)
    elif task.ready():
        result = _json_safe(task.result)
    else:
        result = _json_safe(task.info)

    return JobStatusResponse(
        job_id=job_id,
        status=task.status,
        result=result
    )


