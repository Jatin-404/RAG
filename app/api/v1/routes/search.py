from fastapi import APIRouter
router = APIRouter()

@router.get("/health")
def search_health():
    return {"service": "search", "status": "ok"}
