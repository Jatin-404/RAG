import requests
from app.core.config import settings
import json

PROMPT_TEMPLATE = """You are a helpful assistant. Answer the question using ONLY the context provided below.
If the answer is not in the context, say "I don't have enough information to answer this."

Context:
{context}

Question: {question}

Answer:"""

def generate_answer(question: str, chunks: list[str]) -> str:
    """Non-streaming — used by Celery tasks and reranking pipeline."""
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

def generate_answer_stream(question: str, chunks: list[str]):
    """
    Streaming generator — yields one token at a time from Ollama.
    Each yield is a Server-Sent Event string.
    """
    context = "\n\n".join(chunks)
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)

    response = requests.post(
        f"{settings.OLLAMA_URL}/api/generate",
        json={
            "model": settings.OLLAMA_MODEL,
            "prompt": prompt,
            "stream": True
        },
        stream=True
    )
    response.raise_for_status()

    for line in response.iter_lines():
        if line:
            data = json.loads(line)
            token = data.get("response", "")
            done = data.get("done", False)

            # Send token as SSE
            yield f"data: {json.dumps({'token': token, 'done': done})}\n\n"

            if done:
                break