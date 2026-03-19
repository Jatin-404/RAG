from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.api.v1.routes import ingest, search
from app.db.session import init_db
import uvicorn

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield

app = FastAPI(title="RAG", version="1.0.0", lifespan=lifespan)

app.include_router(ingest.router, prefix="/api/v1/ingest", tags=["ingest"])
app.include_router(search.router, prefix="/api/v1/search", tags=["search"])

@app.get("/health")
def health():
    return {"status": "ok", "version": "1.0.0"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)