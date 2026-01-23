# AI-Powered Document Q&A System (RAG)

An end-to-end **Retrieval-Augmented Generation (RAG)** system that enables users to ask natural-language questions over PDF documents.  
The system retrieves relevant document context using **semantic search** and generates **grounded answers** using a **local Large Language Model (LLM)**.

---

## 🚀 Features

- PDF document ingestion  
- Intelligent text chunking with overlap  
- Semantic embeddings using Hugging Face  
- Fast similarity search with FAISS  
- Retrieval-Augmented Generation (RAG)  
- Local LLM-based answer generation (no paid APIs)  
- Grounded answers with reduced hallucination  
- Modular, production-style architecture  

---

## 🧠 Why RAG?

This approach:

- Reduces hallucinations  
- Improves factual accuracy  
- Scales efficiently to large documents  

---

## 🏗️ System Workflow

1. Load and parse PDF documents  
2. Split text into overlapping chunks  
3. Generate embeddings using Hugging Face models  
4. Store and search embeddings using FAISS  
5. Retrieve relevant context for a user query  
6. Generate a grounded answer using a local LLM  

---

## ⚙️ Tech Stack

- Python  
- Hugging Face Transformers  
- FAISS  
- Local Large Language Model (LLM)  
- Retrieval-Augmented Generation (RAG)  

---

## ▶️ Run the Project

```bash
cd backend
python test_rag.py
