AI-Powered Document Q&A System (RAG)

An end-to-end Retrieval-Augmented Generation (RAG) system that enables users to ask natural-language questions over PDF documents.
The system retrieves relevant document context using semantic search and generates grounded answers using a local LLM.

Features

1. PDF document ingestion

2. Intelligent text chunking with overlap

3. Semantic embeddings using Hugging Face

4. Fast similarity search with FAISS

5. Retrieval-Augmented Generation (RAG)

6. Local LLM answer generation (no paid APIs)

7. Grounded answers with reduced hallucination

8. Modular, production-style architecture


This approach:

Reduces hallucinations

Improves factual accuracy

Scales to large documents

Run the Project
cd backend
python test_rag.py



