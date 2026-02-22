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
MODEL = "text-embedding-3-small"

TOP_K = 3
THRESHOLD = 0.4
CANDIDATE_K = 10  # retrieve 10 first, then filter down to only 3

# Initialize OpenAI client
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not set.")

client = OpenAI(api_key=api_key)

# Load chunks
def load_chunks():
    with open(CHUNK_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

# Load FAISS index
def load_index():
    return faiss.read_index(INDEX_FILE)

# Generate query embedding
def embed_query(query: str) -> np.ndarray:
    response = client.embeddings.create(
        model=MODEL,
        input=query
    )

    vector = np.array(response.data[0].embedding, dtype="float32").reshape(1, -1)

    # Normalize for cosine similarity
    faiss.normalize_L2(vector)
    return vector

# Retrieval core
def retrieve(query: str, index, chunks):
    # Classify (routing decision)
    category, confidence, method = classify_query(query)

    print("\nRouting decision:", category)
    print("Routing method:", method)
    print("Routing confidence:", round(float(confidence), 4))

    # Embed query
    query_vector = embed_query(query)

    # Retrieve candidates
    scores, indices = index.search(query_vector, CANDIDATE_K)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue

        chunk = chunks[idx]
        metadata = chunk.get("metadata", {})
        source_type = metadata.get("source_type", None)

        # Metadata filtering (routing)
        if category is not None and source_type != category:
            continue

        # Threshold filtering (OOD/refusal)
        if float(score) < THRESHOLD:
            continue

        results.append({
            "rank": len(results) + 1,
            "score": float(score),
            "metadata": metadata,
            "chunk_id": int(idx),
            "preview": chunk.get("content", "")[:500],
        })

        if len(results) >= TOP_K:
            break

    return results

# Main
def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/retrieval_strategy.py \"your query\"")
        sys.exit(1)

    query = sys.argv[1]

    print("Loading vector store...")
    chunks = load_chunks()
    index = load_index()

    print("\nQuery:", query)
    print("Top K:", TOP_K)
    print("Threshold:", THRESHOLD)

    results = retrieve(query, index, chunks)

    print("\nResults:")
    print("=" * 80)

    if not results:
        print("No relevant knowledge found (filtered by threshold / routing).")
        return

    for r in results:
        print(f"\nRank {r['rank']}")
        print(f"Score: {r['score']:.4f}")
        print("Metadata:", r["metadata"])
        print("Chunk ID:", r["chunk_id"])
        print("Preview:")
        print(r["preview"])
        print("-" * 80)

if __name__ == "__main__":
    main()
