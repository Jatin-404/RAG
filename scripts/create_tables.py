import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sqlalchemy import text
from app.db.session import engine, init_db, Base
from app.db import models

def main():
    init_db()
    Base.metadata.create_all(bind=engine)

    with engine.connect() as conn:
        # HNSW index for vector search
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw_idx
            ON chunks
            USING hnsw (embedding vector_cosine_ops)
            WITH (m = 16, ef_construction = 64)
        """))
        # Index on document_id for fast cascade deletes and joins
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_chunks_document_id
            ON chunks(document_id)
        """))

        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_chat_messages_session_id
            ON chat_messages(session_id)
        """))
        conn.commit()

    print("Tables and indexes created successfully.")

if __name__ == "__main__":
    main()