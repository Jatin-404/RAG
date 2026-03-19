import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from app.db.session import engine, init_db, Base
from app.db import models  # noqa — registers models with Base
from sqlalchemy import text

def main():
    init_db()
    Base.metadata.create_all(bind=engine)

    # Create HNSW index for fast vector search
    with engine.connect() as conn:
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw_idx
            ON chunks
            USING hnsw (embedding vector_cosine_ops)
            WITH (m = 16, ef_construction = 64)
        """))
        conn.commit()
    
    print("Tables and indexes created successfully.")

if __name__ == "__main__":
    main()