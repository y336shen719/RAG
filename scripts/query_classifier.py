import os
import numpy as np
import faiss
from openai import OpenAI

MODEL = "text-embedding-3-small"

# Lazy OpenAI client
_client = None

def get_client():
    global _client
    if _client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set.")
        _client = OpenAI(api_key=api_key)
    return _client


# Rule-based routing
def rule_based_router(query: str):
    q = query.lower()

    interview_keywords = [
        "conflict", "failure", "strength", "weakness",
        "stress", "leadership", "team", "criticism",
        "tell me about yourself", "self introduction"
    ]

    project_keywords = [
        "pm2.5", "pm25", "kaggle", "model", "pipeline",
        "feature", "evaluation", "metric", "leakage",
        "time series", "lstm", "lasso", "forecast"
    ]

    resume_keywords = [
        "education", "gpa", "experience",
        "skills", "background", "tool",
        "resume", "cv"
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


# Semantic Routing (with caching)
CATEGORY_DESCRIPTIONS = {
    "interview": "behavioral interview questions about teamwork, conflict, leadership, failure, communication",
    "project": "technical machine learning project details, modeling, feature engineering, evaluation, forecasting, time series",
    "resume": "education background, work experience, skills, tools, programming languages, internships"
}

# Global cache
_category_vectors_cache = None


def embed_text(text: str) -> np.ndarray:
    client = get_client()

    response = client.embeddings.create(
        model=MODEL,
        input=text
    )
    vec = np.array(response.data[0].embedding, dtype="float32").reshape(1, -1)
    faiss.normalize_L2(vec)
    return vec


def get_category_vectors():
    global _category_vectors_cache

    if _category_vectors_cache is None:
        client = get_client()

        cache = {}
        for category, description in CATEGORY_DESCRIPTIONS.items():
            response = client.embeddings.create(
                model=MODEL,
                input=description
            )
            vec = np.array(response.data[0].embedding, dtype="float32").reshape(1, -1)
            faiss.normalize_L2(vec)
            cache[category] = vec

        _category_vectors_cache = cache

    return _category_vectors_cache


def semantic_router(query: str):
    query_vec = embed_text(query)
    category_vectors = get_category_vectors()

    best_category = None
    best_score = -1.0

    for category, cat_vec in category_vectors.items():
        score = float((query_vec @ cat_vec.T).item())
        if score > best_score:
            best_score = score
            best_category = category

    return best_category, best_score


# Main classifier
def classify_query(query: str):

    # Rule-based first
    category, confidence = rule_based_router(query)
    if category:
        return category, confidence, "rule"

    # Semantic fallback
    category, score = semantic_router(query)
    return category, score, "semantic"
