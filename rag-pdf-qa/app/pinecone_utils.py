import os

from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

from app.embedding import generate_embedding


# Load environment variables from .env
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

if not PINECONE_API_KEY:
    raise ValueError(
        "PINECONE_API_KEY is not configured. "
        "Add it to your .env file."
    )


pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "pdf-embedding-index"

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-west-2"
        )
    )

index = pc.Index(index_name)


def pdf_already_indexed(pdf_content):
    pdf_text = pdf_content.decode("utf-8", errors="ignore")
    chunks = pdf_text.split()[:100]
    chunk_text = " ".join(chunks)

    vector = generate_embedding(chunk_text)

    query_response = index.query(
        vector=vector.tolist(),
        top_k=1,
        include_metadata=True
    )

    return len(query_response["matches"]) > 0


def index_pdf_chunks(chunks, pdf_hash):
    for i, chunk in enumerate(chunks):
        embedding = generate_embedding(chunk)

        index.upsert(
            vectors=[
                (
                    f"{pdf_hash}_{i}",
                    embedding,
                    {"text": chunk}
                )
            ]
        )


def query_pinecone(query_embedding, top_k=5):
    return index.query(
        vector=query_embedding.tolist(),
        top_k=top_k,
        include_metadata=True
    )