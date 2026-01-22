import faiss
import numpy as np

class FAISSVectorStore:
    def __init__(self, embedding_dim: int):
        self.index = faiss.IndexFlatL2(embedding_dim)
        self.text_chunks = []

    def add_embeddings(self, embeddings: np.ndarray, texts: list[str]):
        """
        Stores embeddings and corresponding text chunks.
        """
        self.index.add(embeddings)
        self.text_chunks.extend(texts)

    def similarity_search(self, query_embedding: np.ndarray, k: int = 3):
        """
        Retrieves top-k most similar text chunks.
        """
        distances, indices = self.index.search(query_embedding, k)
        return [self.text_chunks[i] for i in indices[0]]
