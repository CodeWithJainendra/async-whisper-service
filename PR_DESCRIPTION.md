# Feature: Semantic Audio Search (RAG)

## Summary
Building upon the async architecture, this PR introduces a **Retrieval-Augmented Generation (RAG)** capability for audio. It transforms the service from a simple transcriber into an **Audio Knowledge Base**.

## Key Features
- **Semantic Search**: Users can search through audio recordings using natural language queries (e.g., "Budget discussions" matches audio about "cutting expenses").
- **Vector Embeddings**: integrated `sentence-transformers` to generate 384-dimensional vector embeddings for transcribed segments.
- **Vector Database**: Integrated **ChromaDB** for efficient storage and similarity search of audio chunks.

## Technical Implementation
- **Pipeline**: 
    1.  `Whisper` transcribes audio -> 
    2.  `Tasks` split text into segments -> 
    3.  `SentenceTransformer` encodes segments into vectors -> 
    4.  `ChromaDB` indexes vectors + metadata (start/end timestamps).
- **New Endpoints**:
    - `GET /search?q=...`: Performs cosine similarity search against the indexed audio segments.

## Why this matters
This moves the project into the domain of **AI Engineering**. It demonstrates understanding of:
- Vector Databases & Embeddings
- NLP / Semantic Understanding
- Complex Data Pipelines

## How to Test
1.  Transcribe a file: `POST /asr`
2.  Wait for completion (check `GET /tasks/<id>`).
3.  Search: `GET /search?q=finance`
