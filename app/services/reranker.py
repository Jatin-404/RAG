from sentence_transformers import CrossEncoder

# Loads once at startup
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rerank(query: str, chunks: list[dict], top_k: int = 5) -> list[dict]:
    """
    chunks: list of dicts with at least 'chunk_text' key
    Returns top_k chunks sorted by reranker score
    """
    pairs = [[query, chunk["chunk_text"]] for chunk in chunks]
    scores = reranker.predict(pairs)

    for chunk, score in zip(chunks, scores):
        chunk["rerank_score"] = round(float(score), 4)

    reranked = sorted(chunks, key=lambda x: x["rerank_score"], reverse=True)
    return reranked[:top_k]



'''
Why reranking exists
Your current retrieval does this:
Query → embed → cosine similarity → top 5 chunks
Cosine similarity compares vectors — it finds chunks that are semantically close but
 doesn't deeply understand the relationship between the query and each chunk. 
 It can return chunks that are topically related but don't actually answer the question.
A cross-encoder fixes this:
Query → embed → cosine similarity → top 20 chunks (wide net)
                                          ↓
                        cross-encoder reads query + each chunk together
                                          ↓
                              re-scores all 20, returns top 5
The cross-encoder doesn't use vectors — it reads the query and chunk together as one
 input and scores how well the chunk answers the query. Much more accurate, but too
 slow to run on the entire table — that's why you cast a wide net first with vector 
 search, then rerank the shortlist.
'''