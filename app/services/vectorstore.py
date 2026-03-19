from sqlalchemy.orm import Session
from app.db.models import LegalChunk

def save_chunks(
    db: Session,
    chunks: list[str],
    embeddings: list[list[float]],
    metadata: dict
):
    records = []
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        record = LegalChunk(
            document_id=metadata.get("document_id"),
            filename=metadata.get("filename"),
            chunk_index=i,
            chunk_text=chunk,
            department=metadata.get("department"),
            domain=metadata.get("domain"),
            custom_fields=metadata.get("custom_fields", {}),
            embedding=embedding
        )
        records.append(record)

    db.add_all(records)
    db.commit()
    return len(records)