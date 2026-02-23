# RAG Interview & Project System
This project implements a full Retrieval-Augmented Generation (RAG)
pipeline designed to answer questions about:

-   Interview questions (tech + behaviour) e.g. How to solve conflict at workplace?
-   My machine learning projects
-   Resume and background

## The deployed demo is available at:

### https://ryan-shen-rag-based-interview-assistant.streamlit.app/

For security and controlled access, the application is **password-protected**.

If you are interested in trying the system, please feel free to contact me directly to **request access credentials**.

------------------------------------------------------------------------

## System Architecture Overview

1. User Query

2. Password Gate

3. Query Classifier

-   Rule-based (keyword match)

-   Semantic (embedding similarity)

4. FAISS Vector Search

-   FLAT as index method

-   IP (Inner Product) as similarity search method

6. Two-Stage Retrieval

-   Metadata Filter + Threshold + Top-K

-   Fallback Global Search (if empty)

6. Context Builder

7. LLM Generation

8. Answer

------------------------------------------------------------------------

### 1) Knowledge Base Organization

The knowledge base is structured into three folders:

-   interview_bank/
-   projects/
-   resume/

All files are written in Markdown format.

Frontmatter metadata blocks (— … —) were removed during chunking to
avoid contaminating embeddings with non-semantic information.

------------------------------------------------------------------------

### 2) Chunking (build_chunks.py)

Frontmatter Cleaning: Removes metadata blocks so irrelevant YAML does
not affect embedding quality.

Structured Splitting: - Interview files: split by ### (question-level
granularity) - Projects and Resume: split by # / ## / ###

Minimum Length Filtering: A MIN_CHARS threshold prevents creation of
very short or meaningless chunks.

Output: chunks.json (each chunk contains content + metadata)

------------------------------------------------------------------------

### 3) Embedding + Vector Store (build_embeddings.py)

Each chunk is embedded using an OpenAI embedding model.

Steps: - Generate embeddings - Convert to float32 - Apply L2
normalization - Store in FAISS using IndexFlatIP

Outputs: - embeddings.npy (matrix of vectors) - faiss.index (vector
search index)

------------------------------------------------------------------------

### 4) Retrieval Testing (Early Stage)

retrieval_test.py was used to: - Input query - Specify top_k - Print
similarity scores - Evaluate retrieval quality

------------------------------------------------------------------------

### 5) Query Classifier (query_classifier.py)

Routing Layer determines category:

Rule-Based Router: Keyword matching assigns interview/project/resume
with confidence 0.9.

Semantic Router (Fallback): Embeds query and category descriptions. Uses
cosine similarity to select best category.

Optimization: Category embeddings cached in memory to avoid repeated API
calls.

------------------------------------------------------------------------

### 6) Retrieval Strategy (rag_core.py)

Two-Stage Strategy:

Stage 1: Metadata-Filtered Search - Route to category - Filter by
metadata.source_type - Apply threshold >= 0.2 - Return top_k = 3

Stage 2: Global Fallback If Stage 1 empty: - Run global search - Apply
threshold - Return top_k

------------------------------------------------------------------------

### 7) RAG Generation

rag_core.py: retrieve → build_context → generate_answer

rag_answer.py: Writes output to rag_answer.txt for GitHub Actions
artifact upload.

app.py: Streamlit password-protected interface. Displays generated
answers.

------------------------------------------------------------------------

## Final System Features

-   Vector semantic retrieval
-   Metadata routing
-   Rule-based + semantic classification
-   Threshold filtering (0.2)
-   Top-k control (k=3)
-   Two-stage fallback retrieval
-   Context length control
-   LLM generation with guardrails
-   Streamlit deployment
-   GitHub Actions integration

------------------------------------------------------------------------

## Conclusion

This project demonstrates a production-oriented, retrieval-augmented interview assistant built on a structured personal knowledge base.

The end-to-end classic RAG pipeline is able to deliver concise, context-grounded answers to interview-style questions in real time.
