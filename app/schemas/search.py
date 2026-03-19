from pydantic import BaseModel
from typing import Any

class SearchResult(BaseModel):
    id: int
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
    id: int
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