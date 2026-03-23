from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.v1.routes import ingest, search, documents, chats
from app.db.session import init_db
import uvicorn

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield

app = FastAPI(title="RAG", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(ingest.router, prefix="/api/v1/ingest", tags=["ingest"])
app.include_router(search.router, prefix="/api/v1/search", tags=["search"])
app.include_router(documents.router, prefix="/api/v1/documents", tags=["documents"])
app.include_router(chats.router, prefix="/api/v1/chats", tags=["chats"])

@app.get("/health")
def health():
    return {"status": "ok", "version": "1.0.0"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)