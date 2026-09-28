from fastapi import FastAPI
from pydantic import BaseModel, Field

from app.embedding import generate_embedding
from app.pdf_processing import process_and_index_pdf
from app.pinecone_utils import query_pinecone


app = FastAPI(
    title="PDF RAG API",
    description="Semantic retrieval and RAG API for PDF documents.",
    version="2.0.0"
)


class QueryRequest(BaseModel):
    url: str
    query: str
    top_k: int = Field(default=5, ge=1, le=10)


class RetrievedChunk(BaseModel):
    text: str
    score: float
    chunk_index: int


class QueryResponse(BaseModel):
    query: str
    document_id: str
    results: list[RetrievedChunk]


@app.get("/health")
def health_check():
    return {
        "status": "ok"
    }


@app.post("/query/", response_model=QueryResponse)
def query_pdf(request: QueryRequest):
    """
    Indexes the PDF when necessary and performs semantic
    retrieval against that specific document.
    """
    document_id = process_and_index_pdf(request.url)

    query_embedding = generate_embedding(request.query)

    query_response = query_pinecone(
        query_embedding=query_embedding,
        pdf_hash=document_id,
        top_k=request.top_k
    )

    results = []

    for match in query_response["matches"]:
        metadata = match.get("metadata", {})

        results.append(
            RetrievedChunk(
                text=metadata.get("text", ""),
                score=float(match.get("score", 0.0)),
                chunk_index=int(metadata.get("chunk_index", -1))
            )
        )

    return QueryResponse(
        query=request.query,
        document_id=document_id,
        results=results
    )