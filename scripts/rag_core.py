import os
import json
import numpy as np
import faiss
from openai import OpenAI
from scripts.query_classifier import classify_query

# Configuration
CHUNK_FILE = "vector_store/chunks.json"
INDEX_FILE = "vector_store/faiss.index"

EMBED_MODEL = "text-embedding-3-small"
GEN_MODEL = "gpt-4o-mini"

TOP_K = 3
THRESHOLD = 0.25
CANDIDATE_K = 10
MAX_CONTEXT_CHARS = 6000

# OpenAI Client (Lazy)
def get_client():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set.")
    return OpenAI(api_key=api_key)

# Load Vector Store
def load_vector_store():
    with open(CHUNK_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    index = faiss.read_index(INDEX_FILE)

    return chunks, index

# Embed Query
def embed_query(query: str, client):
    resp = client.embeddings.create(
        model=EMBED_MODEL,
        input=query
    )

    vec = np.array(resp.data[0].embedding, dtype="float32").reshape(1, -1)
    faiss.normalize_L2(vec)

    return vec

# Retrieval (Two-Stage Strategy)
def retrieve(query: str, client, index, chunks):

    category, confidence, method = classify_query(query)

    print("\nRouting decision:", category)
    print("Routing method:", method)
    print("Routing confidence:", round(float(confidence), 4))

    qvec = embed_query(query, client)
    scores, indices = index.search(qvec, CANDIDATE_K)

    # Metadata-filtered search
    results = []

    for score, idx in zip(scores[0], indices[0]):

        if idx == -1:
            continue

        chunk = chunks[idx]
        meta = chunk.get("metadata", {})
        src_type = meta.get("source_type")

        # metadata routing
        if category and src_type != category:
            continue

        # threshold filtering
        if float(score) < THRESHOLD:
            continue

        results.append({
            "score": float(score),
            "chunk_id": int(idx),
            "metadata": meta,
            "content": chunk.get("content", "")
        })

        if len(results) >= TOP_K:
            break

    # Global fallback search
    if not results:

        print("⚠ Metadata filtering returned empty. Running global fallback search.")

        for score, idx in zip(scores[0], indices[0]):

            if idx == -1:
                continue

            if float(score) < THRESHOLD:
                continue

            chunk = chunks[idx]
            meta = chunk.get("metadata", {})

            results.append({
                "score": float(score),
                "chunk_id": int(idx),
                "metadata": meta,
                "content": chunk.get("content", "")
            })

            if len(results) >= TOP_K:
                break

    return results

# Build Context
def build_context(results):

    parts = []
    total_chars = 0

    for i, r in enumerate(results, start=1):

        meta = r["metadata"]
        src_type = meta.get("source_type", "unknown")
        file_name = meta.get("file_name", "unknown")
        tag = meta.get("section") or meta.get("question") or "unknown"

        header = (
            f"[{i}] source_type={src_type} | "
            f"file={file_name} | tag={tag} | "
            f"score={r['score']:.4f}"
        )

        body = r["content"].strip()

        block = header + "\n" + body

        if total_chars + len(block) > MAX_CONTEXT_CHARS:
            break

        parts.append(block)
        total_chars += len(block)

    return "\n\n---\n\n".join(parts)

# Generate Answer
def generate_answer(query: str, context: str, client):

    system_prompt = (
    "You are answering as the job candidate in a professional interview. "
    "Always respond in first-person (e.g., 'I have worked on...', 'In my project...'). "
    "Answer strictly and only based on the provided context. "
    "Do NOT use any external knowledge or fabricate information. "
    "If the relevant information is not found in the context, "
    "respond in first-person by explaining that this is an area I am actively looking to develop "
    "and describe how I plan to strengthen this knowledge or skill in the future. "
    "Maintain a confident, structured, and professional tone."
)

    user_prompt = f"""
Question:
{query}

Context:
{context}

Provide a structured and professional answer.
"""

    response = client.responses.create(
        model=GEN_MODEL,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )

    return response.output_text

# Public API
def answer_query(query: str):

    client = get_client()
    chunks, index = load_vector_store()

    retrieved = retrieve(query, client, index, chunks)

    if not retrieved:
        return "I do not have enough information in the current knowledge base to answer this question."

    context = build_context(retrieved)

    return generate_answer(query, context, client)
