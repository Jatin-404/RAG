from sqlalchemy.orm import Session
from app.db.models import LegalChunk
from pgvector.sqlalchemy import Vector
from sqlalchemy import text

def save_chunks(
    db: Session,
    chunks: list[str],
    embeddings: list[list[float]],
    metadata: dict
):
    records = []
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        # Extract row/summary type from chunk text if present
        chunk_type = "text"
        sheet_name = None
        
        if chunk.startswith("[TABLE_SUMMARY"):
            chunk_type = "table_summary"
            # Extract sheet name from tag
            try:
                sheet_name = chunk.split("sheet=")[1].split("]")[0]
            except Exception:
                pass
            # Store clean text without the tag
            clean_chunk = chunk.split("]\n", 1)[-1]
        elif chunk.startswith("[ROW"):
            chunk_type = "row"
            try:
                sheet_name = chunk.split("sheet=")[1].split("]")[0]
            except Exception:
                pass
            clean_chunk = chunk.split("]\n", 1)[-1]
        else:
            clean_chunk = chunk

        custom_fields = metadata.get("custom_fields", {})
        custom_fields["chunk_type"] = chunk_type
        if sheet_name:
            custom_fields["sheet"] = sheet_name

        record = LegalChunk(
            document_id=metadata.get("document_id"),
            filename=metadata.get("filename"),
            chunk_index=i,
            chunk_text=clean_chunk,
            department=metadata.get("department"),
            domain=metadata.get("domain"),
            custom_fields=custom_fields,
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
    filters = "WHERE 1=1"
    params = {"embedding": str(query_embedding), "top_k": top_k}

    if department:
        filters += " AND department = :department"
        params["department"] = department
    if domain:
        filters += " AND domain = :domain"
        params["domain"] = domain

    sql = text(f"""
        SELECT id, filename, department, domain, chunk_index, chunk_text, custom_fields,
               1 - (embedding <=> CAST(:embedding AS vector)) AS score
        FROM chunks
        {filters}
        ORDER BY embedding <=> CAST(:embedding AS vector)
        LIMIT :top_k
    """)

    results = db.execute(sql, params).fetchall()
    return results