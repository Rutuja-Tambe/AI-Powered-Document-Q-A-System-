from sentence_transformers import SentenceTransformer

class EmbeddingModel:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_texts(self, texts: list[str]):
        """
        Generates embeddings for a list of texts.
        """
        return self.model.encode(texts, convert_to_numpy=True)

    def embed_query(self, query: str):
        """
        Generates embedding for a user query.
        """
        return self.model.encode([query], convert_to_numpy=True)
