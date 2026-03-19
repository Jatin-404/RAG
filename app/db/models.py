# What does the legal_chunks table look like
# You run this once manually. After that, the table exists permanently in your DB.

from sqlalchemy import Column, String, Integer, DateTime, JSON, func
from pgvector.sqlalchemy import Vector
from app.db.session import Base

class LegalChunk(Base):
    __tablename__ = "chunks"

    id                  = Column(Integer, primary_key=True, autoincrement=True)
    document_id         = Column(String, nullable=False)
    judgment_id         = Column(String, nullable=False)
    filename            = Column(String)
    chunk_index         = Column(Integer)
    chunk_text          = Column(String, nullable=False)
    court               = Column(String)
    court_level         = Column(String)
    decision_date       = Column(String)
    domain              = Column(String)
    bench               = Column(String)
    jurisdiction        = Column(String)
    ingestion_timestamp = Column(DateTime, server_default=func.now())
    custom_fields       = Column(JSON, default={})
    embedding           = Column(Vector(1024))