from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from app.api.v1.dependencies import get_db
from app.services.embedder import embed_chunks
from app.services.vectorstore import search_chunks
from app.services.rag import generate_answer
from app.services.reranker import rerank

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
    # Step 1: embed query
    query_embedding = embed_chunks([q])[0]

    # Step 2: wide net — retrieve top 20
    results = search_chunks(db, query_embedding, top_k=20, department=department, domain=domain)

    # Step 3: rerank — return best 5
    candidates = [
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
    reranked = rerank(q, candidates, top_k=top_k)

    # Step 4: generate answer from reranked chunks
    chunks_text = [r["chunk_text"] for r in reranked]
    answer = generate_answer(q, chunks_text)

    return {
        "question": q,
        "answer": answer,
        "sources": reranked
    }