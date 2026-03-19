from pydantic import BaseModel
from typing import Any

class IngestResponse(BaseModel):
    status: str
    filename: str
    chunks_stored: int

class QueuedResponse(BaseModel):
    status: str
    job_id: str
    filename: str

class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    result: Any | None = None