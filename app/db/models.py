# What does the legal_chunks table look like
# You run this once manually. After that, the table exists permanently in your DB.

# app/db/models.py
import uuid
from sqlalchemy import Column, String, Integer, DateTime, JSON, Boolean, func, ForeignKey, text
from sqlalchemy.dialects.postgresql import UUID
from pgvector.sqlalchemy import Vector
from app.db.session import Base

class Document(Base):
    __tablename__ = "documents"

    id              = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    filename        = Column(String, nullable=False)
    department      = Column(String)
    domain          = Column(String)
    chunk_count     = Column(Integer, default=0)
    created_at      = Column(DateTime, server_default=func.now())
    metadata_fields = Column(JSON, default={})
    is_deleted      = Column(Boolean, default=False)

class Chunk(Base):
    __tablename__ = "chunks"

    id          = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    filename    = Column(String)
    chunk_index = Column(Integer)
    chunk_text  = Column(String, nullable=False)
    department  = Column(String)
    domain      = Column(String)
    chunk_type  = Column(String, default="text")
    custom_fields = Column(JSON, default={})
    embedding   = Column(Vector(1024))

class ChatSession(Base):
    __tablename__ = "chat_sessions"

    id         = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title      = Column(String, default="New Chat")
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id         = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(UUID(as_uuid=True), ForeignKey("chat_sessions.id", ondelete="CASCADE"), nullable=False)
    role       = Column(String, nullable=False)  # "user" or "ai"
    content    = Column(String, nullable=False)
    sources    = Column(JSON, default=[])
    created_at = Column(DateTime, server_default=func.now())