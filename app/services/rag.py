import requests
from app.core.config import settings

PROMPT_TEMPLATE = """You are a helpful assistant. Answer the question using ONLY the context provided below.
If the answer is not in the context, say "I don't have enough information to answer this."

Context:
{context}

Question: {question}

Answer:"""

def generate_answer(question: str, chunks: list[str]) -> str:
    context = "\n\n".join(chunks)
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)

    response = requests.post(
        f"{settings.OLLAMA_URL}/api/generate",
        json={
            "model": settings.OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False
        }
    )
    response.raise_for_status()
    return response.json()["response"]