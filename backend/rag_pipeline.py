import os
from ingestion import extract_text_from_pdf
from chunking import chunk_text
from embeddings import EmbeddingModel
from vector_store import FAISSVectorStore
from llm import generate_answer


class RAGPipeline:
    """
    End-to-end Retrieval-Augmented Generation (RAG) pipeline.
    """

    def __init__(self):
        self.embedder = EmbeddingModel()
        self.vector_store = None

    def ingest_document(self, pdf_path: str):
        """
        Ingest a PDF document into the vector store.
        """

        # Convert relative path to absolute project path
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        pdf_path = os.path.join(project_root, pdf_path)

        # Step 1: Extract text
        text = extract_text_from_pdf(pdf_path)

        # Step 2: Chunk text
        chunks = chunk_text(text)

        # Step 3: Embed chunks
        embeddings = self.embedder.embed_texts(chunks)

        # Step 4: Store in FAISS
        self.vector_store = FAISSVectorStore(
            embedding_dim=embeddings.shape[1]
        )
        self.vector_store.add_embeddings(embeddings, chunks)

    def retrieve_context(self, query: str, k: int = 2):
        """
        Retrieve top-k relevant chunks.
        """
        if self.vector_store is None:
            raise ValueError("Document not ingested yet.")

        query_embedding = self.embedder.embed_query(query)
        return self.vector_store.similarity_search(query_embedding, k)

    def answer(self, query: str):
        """
        Generate grounded answer using retrieved context.
        """
        context_chunks = self.retrieve_context(query)

        # Debug (can remove later)
        print("----- RETRIEVED CONTEXT -----")
        for i, chunk in enumerate(context_chunks):
            print(f"[Chunk {i+1}]:\n{chunk[:400]}\n")
        print("----- END CONTEXT -----")

        context_text = "\n\n".join(context_chunks)

        prompt = f"""
You are a helpful assistant.

Answer the question ONLY using the information below.
If the answer is not clearly mentioned, say "Not found in document".

Context:
{context_text}

Question:
{query}

Answer:
"""

        return generate_answer(prompt)
