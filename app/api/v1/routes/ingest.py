from fastapi import APIRouter
router = APIRouter()

@router.get("/health")
def ingest_health():
    return {"service": "ingest", "status": "ok"}