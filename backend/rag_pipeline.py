from ingestion import extract_text_from_pdf
from chunking import chunk_text
from embeddings import EmbeddingModel
from vector_store import FAISSVectorStore
from llm import generate_answer


class RAGPipeline:
    """
    End-to-end Retrieval-Augmented Generation (RAG) pipeline.
    Handles document ingestion, retrieval, and answer generation.
    """

    def __init__(self):
        self.embedder = EmbeddingModel()
        self.vector_store = None

    def ingest_document(self, pdf_path: str):
        """
        Ingests a PDF document:
        1. Extracts text
        2. Chunks text
        3. Generates embeddings
        4. Stores embeddings in FAISS
        """
        # Step 1: Extract text from PDF
        text = extract_text_from_pdf(pdf_path)

        # Step 2: Chunk the text
        chunks = chunk_text(text)

        # Step 3: Generate embeddings
        embeddings = self.embedder.embed_texts(chunks)

        # Step 4: Store embeddings in FAISS
        self.vector_store = FAISSVectorStore(
            embedding_dim=embeddings.shape[1]
        )
        self.vector_store.add_embeddings(embeddings, chunks)

    def retrieve_context(self, query: str, k: int = 3):
        """
        Retrieves top-k relevant chunks for a query.
        """
        if self.vector_store is None:
            raise ValueError("No document ingested. Please ingest a document first.")

        query_embedding = self.embedder.embed_query(query)
        return self.vector_store.similarity_search(query_embedding, k)

    def answer(self, query: str):
        """
        Generates a grounded answer using retrieved context and LLM.
        """
        context_chunks = self.retrieve_context(query)

        context_text = "\n\n".join(context_chunks)

        prompt = f"""
You are an AI assistant. Answer the question using ONLY the context below.
If the answer is not present in the context, say "Not found in document".

Context:
{context_text}

Question:
{query}
"""

        return generate_answer(prompt)
