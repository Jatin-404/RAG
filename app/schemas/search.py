from pydantic import BaseModel
from typing import Any
from uuid import UUID

class SearchResult(BaseModel):
    id: UUID
    filename: str
    department: str | None
    domain: str | None
    chunk_text: str
    custom_fields: dict[str, Any]
    score: float

class SearchResponse(BaseModel):
    query: str
    results: list[SearchResult]

class RankedResult(BaseModel):
    id: UUID
    filename: str
    department: str | None
    domain: str | None
    chunk_text: str
    custom_fields: dict[str, Any]
    score: float
    rerank_score: float

class AskResponse(BaseModel):
    question: str
    answer: str
    sources: list[RankedResult]