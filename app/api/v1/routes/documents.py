from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import text
from app.api.v1.dependencies import get_db
from app.services.vectorstore import delete_document, hard_delete_document

router = APIRouter()

@router.get("/")
def list_documents(db: Session = Depends(get_db)):
    rows = db.execute(text("""
        SELECT 
            id::text as document_id,
            filename,
            department,
            domain,
            chunk_count,
            created_at,
            metadata_fields,
            is_deleted
        FROM documents
        WHERE is_deleted = FALSE
        ORDER BY created_at DESC
    """)).fetchall()

    return {
        "documents": [
            {
                "document_id": row.document_id,
                "filename": row.filename,
                "department": row.department,
                "domain": row.domain,
                "chunk_count": row.chunk_count,
                "created_at": row.created_at,
                "metadata_fields": row.metadata_fields
            }
            for row in rows
        ]
    }

@router.delete("/{document_id}")
def delete_doc(document_id: str, hard: bool = False, db: Session = Depends(get_db)):
    if hard:
        result = hard_delete_document(db, document_id)
    else:
        result = delete_document(db, document_id)

    if not result:
        raise HTTPException(status_code=404, detail="Document not found")

    return {
        "status": "deleted",
        "document_id": document_id,
        "type": "hard" if hard else "soft"
    }