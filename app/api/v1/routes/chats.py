import uuid
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import text
from app.api.v1.dependencies import get_db
from app.db.models import ChatSession, ChatMessage
from app.services.embedder import embed_chunks
from app.services.vectorstore import search_chunks
from app.services.rag import generate_answer
from app.services.reranker import rerank
from pydantic import BaseModel

router = APIRouter()

class AskInChat(BaseModel):
    question: str
    top_k: int = 5
    department: str | None = None
    domain: str | None = None

# ── List all sessions ──────────────────────────────────────────────
@router.get("/")
def list_sessions(db: Session = Depends(get_db)):
    rows = db.execute(text("""
        SELECT id::text, title, created_at, updated_at
        FROM chat_sessions
        ORDER BY updated_at DESC
    """)).fetchall()
    return {"sessions": [
        {"id": r.id, "title": r.title, "created_at": r.created_at, "updated_at": r.updated_at}
        for r in rows
    ]}

# ── Create new session ─────────────────────────────────────────────
@router.post("/")
def create_session(db: Session = Depends(get_db)):
    session = ChatSession()
    db.add(session)
    db.commit()
    db.refresh(session)
    return {"id": str(session.id), "title": session.title, "created_at": session.created_at}

# ── Get messages for a session ─────────────────────────────────────
@router.get("/{session_id}/messages")
def get_messages(session_id: str, db: Session = Depends(get_db)):
    rows = db.execute(text("""
        SELECT id::text, role, content, sources, created_at
        FROM chat_messages
        WHERE session_id = CAST(:session_id AS uuid)
        ORDER BY created_at ASC
    """), {"session_id": session_id}).fetchall()
    return {"messages": [
        {"id": r.id, "role": r.role, "content": r.content, "sources": r.sources, "created_at": r.created_at}
        for r in rows
    ]}

# ── Ask a question inside a session ───────────────────────────────
@router.post("/{session_id}/ask")
def ask_in_session(session_id: str, body: AskInChat, db: Session = Depends(get_db)):
    # Verify session exists
    session = db.execute(text(
        "SELECT id, title FROM chat_sessions WHERE id = CAST(:id AS uuid)"
    ), {"id": session_id}).fetchone()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # Save user message
    user_msg = ChatMessage(
        session_id=uuid.UUID(session_id),
        role="user",
        content=body.question,
        sources=[]
    )
    db.add(user_msg)

    # RAG pipeline
    query_embedding = embed_chunks([body.question])[0]
    results = search_chunks(db, query_embedding, top_k=20,
                            department=body.department, domain=body.domain)
    candidates = [
        {
            "id": str(row.id),
            "filename": row.filename,
            "department": row.department,
            "domain": row.domain,
            "chunk_text": row.chunk_text,
            "custom_fields": row.custom_fields,
            "score": round(row.score, 4)
        }
        for row in results
    ]
    reranked = rerank(body.question, candidates, top_k=body.top_k)
    chunks_text = [r["chunk_text"] for r in reranked]
    answer = generate_answer(body.question, chunks_text)

    # Save AI message
    ai_msg = ChatMessage(
        session_id=uuid.UUID(session_id),
        role="ai",
        content=answer,
        sources=reranked
    )
    db.add(ai_msg)

    # Update session title from first question
    if session.title == "New Chat":
        title = body.question[:60] + ("..." if len(body.question) > 60 else "")
        db.execute(text(
            "UPDATE chat_sessions SET title = :title, updated_at = NOW() WHERE id = CAST(:id AS uuid)"
        ), {"title": title, "id": session_id})
    else:
        db.execute(text(
            "UPDATE chat_sessions SET updated_at = NOW() WHERE id = CAST(:id AS uuid)"
        ), {"id": session_id})

    db.commit()

    return {
        "answer": answer,
        "sources": reranked,
        "session_id": session_id
    }

# ── Delete a session ───────────────────────────────────────────────
@router.delete("/{session_id}")
def delete_session(session_id: str, db: Session = Depends(get_db)):
    result = db.execute(text("""
        DELETE FROM chat_sessions
        WHERE id = CAST(:id AS uuid)
        RETURNING id
    """), {"id": session_id}).fetchone()
    db.commit()
    if not result:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"status": "deleted", "session_id": session_id}