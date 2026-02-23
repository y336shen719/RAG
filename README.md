# RAG Interview & Project Assistant

This project implements a full Retrieval-Augmented Generation (RAG)
pipeline designed to answer questions about:

-   Interview experiences
-   Machine learning projects
-   Resume and background

------------------------------------------------------------------------

## System Architecture Overview

User Query | v [Password Gate (Streamlit)] | v
rag_core.answer_query(query) | v query_classifier.classify_query(query)
|– (Rule-based router: keyword match) |– else (Semantic router:
embeddings) | v FAISS retrieval (index.search) | v Retrieval Strategy
(Two-stage) |– Stage 1: metadata filter + threshold + top_k |– if empty
-> Stage 2: global fallback | v Context Builder | v LLM Answer
Generation | v Final Answer

### The entire RAG pipeline looks like:
User Query
   |
   v
[Password Gate (Streamlit)]
   |
   v
rag_core.answer_query(query)
   |
   v
query_classifier.classify_query(query)
   |-- (Rule-based router: keyword match)
   |-- else (Semantic router: embeddings)
   |
   v
FAISS retrieval (index.search)
   |
   v
Retrieval Strategy (Two-stage)
   |-- Stage 1: metadata filter + threshold + top_k
   |-- if empty -> Stage 2: global fallback
   |
   v
Context Builder
   |
   v
LLM Answer Generation
   |
   v
Final Answer

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

This project implements a Classic RAG System. It is modular and can be
extended into a full Agent architecture.
