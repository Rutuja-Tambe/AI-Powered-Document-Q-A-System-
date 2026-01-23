import faiss
import numpy as np


class FAISSVectorStore:
    """
    FAISS vector store for similarity search.
    """

    def __init__(self, embedding_dim: int):
        self.index = faiss.IndexFlatL2(embedding_dim)
        self.text_chunks = []

    def add_embeddings(self, embeddings: np.ndarray, texts):
        self.index.add(embeddings)
        self.text_chunks.extend(texts)

    def similarity_search(self, query_embedding: np.ndarray, k: int = 2):
        distances, indices = self.index.search(query_embedding, k)
        return [self.text_chunks[i] for i in indices[0]]
