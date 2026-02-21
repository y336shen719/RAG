import os
import json
import numpy as np
import faiss
from openai import OpenAI
from tqdm import tqdm

CHUNK_FILE = "chunks.json"
EMBEDDING_OUTPUT = "embeddings.npy"
INDEX_OUTPUT = "faiss.index"

MODEL = "text-embedding-3-small"
BATCH_SIZE = 32

# ----------------------------
# Initialize OpenAI client
# ----------------------------
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not set in environment.")

client = OpenAI(api_key=api_key)


# ----------------------------
# Load chunks
# ----------------------------
def load_chunks():
    with open(CHUNK_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


# ----------------------------
# Generate embeddings (batched)
# ----------------------------
def generate_embeddings(chunks):
    texts = [chunk["content"] for chunk in chunks]
    vectors = []

    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Generating embeddings"):
        batch_texts = texts[i:i + BATCH_SIZE]

        response = client.embeddings.create(
            model=MODEL,
            input=batch_texts
        )

        for item in response.data:
            vectors.append(item.embedding)

    vectors = np.array(vectors).astype("float32")

    # Normalize for cosine similarity
    faiss.normalize_L2(vectors)

    return vectors


# ----------------------------
# Build FAISS index (cosine similarity)
# ----------------------------
def build_faiss_index(vectors):
    dim = vectors.shape[1]

    # Inner product + normalized vectors ≈ cosine similarity
    index = faiss.IndexFlatIP(dim)     # index method: flat; similarity measure method: IP (inner product)
    index.add(vectors)

    return index


# ----------------------------
# Main pipeline
# ----------------------------
def main():
    print("Loading chunks...")
    chunks = load_chunks()

    if len(chunks) == 0:
        print("No chunks found.")
        return

    print("Generating embeddings...")
    vectors = generate_embeddings(chunks)

    print("Saving embeddings...")
    np.save(EMBEDDING_OUTPUT, vectors)

    print("Building FAISS index...")
    index = build_faiss_index(vectors)

    print("Saving FAISS index...")
    faiss.write_index(index, INDEX_OUTPUT)

    print("Done.")


if __name__ == "__main__":
    main()
