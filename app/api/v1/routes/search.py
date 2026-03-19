from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from app.api.v1.dependencies import get_db
from app.services.embedder import embed_chunks
from app.services.vectorstore import search_chunks

router = APIRouter()

@router.get("/health")
def search_health():
    return {"service": "search", "status": "ok"}

@router.get("/")
def search(
    q: str = Query(..., description="Search query"),
    top_k: int = Query(5, description="Number of results"),
    department: str = Query(None, description="Filter by department"),
    domain: str = Query(None, description="Filter by domain"),
    db: Session = Depends(get_db)
):
    query_embedding = embed_chunks([q])[0]

    results = search_chunks(
        db=db,
        query_embedding=query_embedding,
        top_k=top_k,
        department=department,
        domain=domain
    )

    return {
        "query": q,
        "results": [
            {
                "id": row.id,
                "filename": row.filename,
                "department": row.department,
                "domain": row.domain,
                "chunk_text": row.chunk_text,
                "custom_fields": row.custom_fields,
                "score": round(row.score, 4)
            }
            for row in results
        ]
    }
