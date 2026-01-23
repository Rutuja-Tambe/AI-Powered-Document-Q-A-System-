from rag_pipeline import RAGPipeline

rag = RAGPipeline()

# Ingest sample document
rag.ingest_document("data/sample.pdf")

# Ask question
query = "What is this document about?"
answer = rag.answer(query)

print("\nAnswer:")
print(answer)
