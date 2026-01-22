from rag_pipeline import RAGPipeline

rag = RAGPipeline()
rag.ingest_document("data/sample.pdf")

query = "What is this document about?"
answer = rag.answer(query)

print("Answer:")
print(answer)
