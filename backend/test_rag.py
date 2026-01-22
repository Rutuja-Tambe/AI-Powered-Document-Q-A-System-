from rag_pipeline import RAGPipeline

rag = RAGPipeline()
rag.ingest_document("data/sample.pdf")

query = "What is this document about?"
context = rag.answer(query)

print(context)
