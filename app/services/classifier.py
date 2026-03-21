import json
import requests
from app.core.config import settings

CLASSIFY_PROMPT = """You are a document classifier for a company's internal document system.

Analyze the following document excerpt and return a JSON object with these fields:
- department: one of [hr, legal, finance, engineering, operations, general]
- domain: a short specific label like leave_policy, contract, invoice, onboarding, rfc, budget
- custom_fields: a dict of any relevant metadata you can extract (document_type, topics, entities, dates mentioned)

Return ONLY valid JSON. No explanation, no markdown, no extra text.

Document excerpt:
{text}

JSON:"""

def classify_document(text: str) -> dict:
    # Use first 1000 chars — enough context, not too slow
    excerpt = text[:1000]
    prompt = CLASSIFY_PROMPT.format(text=excerpt)

    response = requests.post(
        f"{settings.OLLAMA_URL}/api/generate",
        json={
            "model": settings.OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False
        }
    )
    response.raise_for_status()
    raw = response.json()["response"].strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # LLM occasionally adds extra text — fallback to safe defaults
        return {
            "department": "general",
            "domain": "general",
            "custom_fields": {}
        }