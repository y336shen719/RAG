import os
import sys
import json
import numpy as np
import faiss
from openai import OpenAI
from query_classifier import classify_query

# Configuration
CHUNK_FILE = "vector_store/chunks.json"
INDEX_FILE = "vector_store/faiss.index"

EMBED_MODEL = "text-embedding-3-small"
GEN_MODEL = "gpt-4o-mini"

TOP_K = 3
THRESHOLD = 0.4
CANDIDATE_K = 10
MAX_CONTEXT_CHARS = 6000

# Initialize OpenAI client
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not set.")

client = OpenAI(api_key=api_key)

# Load chunks & index
def load_chunks():
    with open(CHUNK_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def load_index():
    return faiss.read_index(INDEX_FILE)

# Embedding
def embed_query(query: str) -> np.ndarray:
    resp = client.embeddings.create(
        model=EMBED_MODEL,
        input=query
    )
    vec = np.array(resp.data[0].embedding, dtype="float32").reshape(1, -1)
    faiss.normalize_L2(vec)
    return vec

# Retrieval
def retrieve(query: str, index, chunks):
    category, confidence, method = classify_query(query)

    print("\nRouting decision:", category)
    print("Routing method:", method)
    print("Routing confidence:", round(float(confidence), 4))

    qvec = embed_query(query)
    scores, indices = index.search(qvec, CANDIDATE_K)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue

        chunk = chunks[idx]
        meta = chunk.get("metadata", {})
        src_type = meta.get("source_type")

        if category is not None and src_type != category:
            continue

        if float(score) < THRESHOLD:
            continue

        results.append({
            "rank": len(results) + 1,
            "score": float(score),
            "chunk_id": int(idx),
            "metadata": meta,
            "content": chunk.get("content", "")
        })

        if len(results) >= TOP_K:
            break

    return results

# Context builder
def build_context(results):
    parts = []
    total = 0

    for i, r in enumerate(results, start=1):
        meta = r["metadata"]
        src_type = meta.get("source_type", "unknown")
        file_name = meta.get("file_name", "unknown")
        tag = meta.get("section") or meta.get("question") or "unknown"

        header = f"[{i}] source_type={src_type} | file={file_name} | tag={tag} | score={r['score']:.4f} | chunk_id={r['chunk_id']}"
        body = r["content"].strip()

        block = header + "\n" + body

        if total + len(block) > MAX_CONTEXT_CHARS:
            break

        parts.append(block)
        total += len(block)

    return "\n\n---\n\n".join(parts)

# Prompt Template
def build_messages(query: str, context: str):

    system = (
        "You are a professional interview and project assistant. "
        "Answer ONLY using the provided context. "
        "If the context does not contain enough information, say you do not have enough information. "
        "Be structured, clear, and professional. "
        "Cite sources using bracket notation like [1], [2]."
    )

    user = (
        f"User Question:\n{query}\n\n"
        f"Context:\n{context}\n\n"
        "Instructions:\n"
        "- Provide a clear and structured answer.\n"
        "- Use citations like [1], [2] when referencing context.\n"
        "- Do NOT invent information outside the context.\n"
    )

    return system, user

# Generation
def generate_answer(system_msg: str, user_msg: str):

    resp = client.responses.create(
        model=GEN_MODEL,
        input=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.2,
    )

    return resp.output_text

# Main
def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/rag_answer.py \"your question\"")
        sys.exit(1)

    query = sys.argv[1]

    print("Loading vector store...")
    chunks = load_chunks()
    index = load_index()

    print("\nQuery:", query)
    print("TOP_K:", TOP_K)
    print("THRESHOLD:", THRESHOLD)

    retrieved = retrieve(query, index, chunks)

    if not retrieved:
        print("\nNo relevant knowledge found.")
        print("I do not have enough information in the current knowledge base to answer this question.")
        return

    context = build_context(retrieved)
    system_msg, user_msg = build_messages(query, context)

    print("\nGenerating answer...")
    answer = generate_answer(system_msg, user_msg)

    print("\n" + "=" * 80)
    print(answer)
    print("=" * 80)

if __name__ == "__main__":
    main()
