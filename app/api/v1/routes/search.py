from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from app.api.v1.dependencies import get_db
from app.services.embedder import embed_chunks
from app.services.vectorstore import search_chunks
from app.services.rag import generate_answer

router = APIRouter()

@router.get("/health")
def search_health():
    return {"service": "search", "status": "ok"}

@router.get("/")
def search(
    q: str = Query(...),
    top_k: int = Query(5),
    department: str = Query(None),
    domain: str = Query(None),
    db: Session = Depends(get_db)
):
    query_embedding = embed_chunks([q])[0]
    results = search_chunks(db, query_embedding, top_k, department, domain)

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

@router.get("/ask")
def ask(
    q: str = Query(...),
    top_k: int = Query(5),
    department: str = Query(None),
    domain: str = Query(None),
    db: Session = Depends(get_db)
):
    # Step 1: embed the question
    query_embedding = embed_chunks([q])[0]

    # Step 2: retrieve relevant chunks
    results = search_chunks(db, query_embedding, top_k, department, domain)
    chunks = [row.chunk_text for row in results]

    # Step 3: generate answer from chunks
    answer = generate_answer(q, chunks)

    return {
        "question": q,
        "answer": answer,
        "sources": [
            {
                "filename": row.filename,
                "chunk_text": row.chunk_text,
                "score": round(row.score, 4)
            }
            for row in results
        ]
    }