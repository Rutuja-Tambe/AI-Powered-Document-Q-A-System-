AI-Powered Document Q&A System (RAG)

An end-to-end Retrieval-Augmented Generation (RAG) system that enables users to ask natural-language questions over PDF documents.
The system retrieves relevant document context using semantic search and generates grounded answers using a local LLM.

Features

📑 PDF document ingestion

✂️ Intelligent text chunking with overlap

🔢 Semantic embeddings using Hugging Face

⚡ Fast similarity search with FAISS

🧠 Retrieval-Augmented Generation (RAG)

💬 Local LLM answer generation (no paid APIs)

🔐 Grounded answers with reduced hallucination

🧩 Modular, production-style architecture


This approach:

Reduces hallucinations

Improves factual accuracy

Scales to large documents

Run the Project
cd backend
python test_rag.py


