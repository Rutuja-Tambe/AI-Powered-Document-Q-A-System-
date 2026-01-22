from ingestion import extract_text_from_pdf
from chunking import chunk_text
from embeddings import EmbeddingModel
from vector_store import FAISSVectorStore

class RAGPipeline:
    def __init__(self):
        self.embedder = EmbeddingModel()
        self.vector_store = None

    def ingest_document(self, pdf_path: str):
        # Step 1: Extract text
        text = extract_text_from_pdf(pdf_path)

        # Step 2: Chunk text
        chunks = chunk_text(text)

        # Step 3: Create embeddings
        embeddings = self.embedder.embed_texts(chunks)

        # Step 4: Store in FAISS
        self.vector_store = FAISSVectorStore(
            embedding_dim=embeddings.shape[1]
        )
        self.vector_store.add_embeddings(embeddings, chunks)

    def retrieve_context(self, query: str, k: int = 3):
        query_embedding = self.embedder.embed_query(query)
        return self.vector_store.similarity_search(query_embedding, k)

    def answer(self, query: str):
        context = self.retrieve_context(query)
        return context
