from sqlalchemy.orm import Session
from sqlalchemy import text
from app.db.models import Document, Chunk
import uuid

def save_chunks(
    db: Session,
    chunks: list[str],
    embeddings: list[list[float]],
    metadata: dict
):
    document_id = uuid.UUID(metadata.get("document_id")) if isinstance(metadata.get("document_id"), str) else metadata.get("document_id")

    # 1. Insert document record first
    doc = Document(
        id=document_id,
        filename=metadata.get("filename"),
        department=metadata.get("department"),
        domain=metadata.get("domain"),
        chunk_count=len(chunks),
        metadata_fields=metadata.get("custom_fields", {})
    )
    db.add(doc)
    db.flush()  # write document before chunks (FK constraint)

    # 2. Insert chunks with FK reference
    records = []
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        # Parse chunk type from tag
        chunk_type = "text"
        if chunk.startswith("[TABLE_SUMMARY"):
            chunk_type = "table_summary"
        elif chunk.startswith("[ROW"):
            chunk_type = "row"
        elif chunk.startswith("[TEXT]"):
            chunk_type = "text"
        clean_chunk = chunk.split("]\n", 1)[-1] if "]" in chunk else chunk

        record = Chunk(
            id=uuid.uuid4(),
            document_id=document_id,
            filename=metadata.get("filename"),
            chunk_index=i,
            chunk_text=clean_chunk,
            department=metadata.get("department"),
            domain=metadata.get("domain"),
            chunk_type=chunk_type,
            custom_fields=metadata.get("custom_fields", {}),
            embedding=embedding
        )
        records.append(record)

    db.add_all(records)
    db.commit()
    return len(records)


def search_chunks(
    db: Session,
    query_embedding: list[float],
    top_k: int = 5,
    department: str = None,
    domain: str = None
):
    filters = "WHERE c.document_id IN (SELECT id FROM documents WHERE is_deleted = FALSE)"
    params = {"embedding": str(query_embedding), "top_k": top_k}

    if department:
        filters += " AND c.department = :department"
        params["department"] = department
    if domain:
        filters += " AND c.domain = :domain"
        params["domain"] = domain

    sql = text(f"""
        SELECT c.id, c.filename, c.department, c.domain, c.chunk_index,
               c.chunk_text, c.custom_fields, c.chunk_type,
               1 - (c.embedding <=> CAST(:embedding AS vector)) AS score
        FROM chunks c
        {filters}
        ORDER BY c.embedding <=> CAST(:embedding AS vector)
        LIMIT :top_k
    """)

    return db.execute(sql, params).fetchall()


def delete_document(db: Session, document_id: str):
    """
    Soft delete — marks document as deleted.
    Chunks remain in DB but excluded from search.
    Hard delete also provided for full cleanup.
    """
    sql = text("""
        UPDATE documents 
        SET is_deleted = TRUE 
        WHERE id = CAST(:document_id AS uuid)
        RETURNING id, filename
    """)
    result = db.execute(sql, {"document_id": document_id}).fetchone()
    db.commit()
    return result


def hard_delete_document(db: Session, document_id: str):
    """
    Hard delete — removes document and all chunks via CASCADE.
    """
    sql = text("""
        DELETE FROM documents 
        WHERE id = CAST(:document_id AS uuid)
        RETURNING id, filename
    """)
    result = db.execute(sql, {"document_id": document_id}).fetchone()
    db.commit()
    return result