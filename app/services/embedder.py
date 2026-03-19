from langchain_huggingface import HuggingFaceEmbeddings
from app.core.config import settings

embedder = HuggingFaceEmbeddings(
    model_name=settings.EMBED_MODEL,
    model_kwargs={"device": "cpu"}
)

def embed_chunks(chunks: list[str]) -> list[list[float]]:
    return embedder.embed_documents(chunks)



# module-level embedder object so the model loads once when the service starts — not on every request