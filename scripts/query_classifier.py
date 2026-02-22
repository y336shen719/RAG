import os
import numpy as np
from openai import OpenAI
import faiss

MODEL = "text-embedding-3-small"

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not set.")

client = OpenAI(api_key=api_key)

# Rule-based routing
def rule_based_router(query: str):
    q = query.lower()

    interview_keywords = [
        "conflict", "failure", "strength", "weakness",
        "stress", "leadership", "team", "criticism"
    ]

    project_keywords = [
        "pm2.5", "kaggle", "model", "pipeline",
        "feature", "evaluation", "metric", "leakage"
    ]

    resume_keywords = [
        "education", "gpa", "experience",
        "skills", "background", "tool"
    ]

    for word in interview_keywords:
        if word in q:
            return "interview", 0.9

    for word in project_keywords:
        if word in q:
            return "project", 0.9

    for word in resume_keywords:
        if word in q:
            return "resume", 0.9

    return None, 0.0

# Semantic routing fallback
def embed_text(text):
    response = client.embeddings.create(
        model=MODEL,
        input=text
    )
    vec = np.array(response.data[0].embedding, dtype="float32").reshape(1, -1)
    faiss.normalize_L2(vec)
    return vec


CATEGORY_DESCRIPTIONS = {
    "interview": "behavioral interview questions about teamwork, conflict, leadership, failure",
    "project": "technical machine learning project details, modeling, evaluation, time series",
    "resume": "education background, work experience, skills, internships"
}

# Precompute category embeddings once
CATEGORY_VECTORS = {
    k: embed_text(v) for k, v in CATEGORY_DESCRIPTIONS.items()
}

def semantic_router(query: str):
    query_vec = embed_text(query)

    best_category = None
    best_score = -1

    for category, cat_vec in CATEGORY_VECTORS.items():
        score = float(np.dot(query_vec, cat_vec.T))
        if score > best_score:
            best_score = score
            best_category = category

    return best_category, best_score


# ----------------------------
# Main classification function
# ----------------------------
def classify_query(query: str):

    # Rule-based first
    category, confidence = rule_based_router(query)
    if category:
        return category, confidence, "rule"

    # Semantic fallback
    category, score = semantic_router(query)
    return category, score, "semantic"
