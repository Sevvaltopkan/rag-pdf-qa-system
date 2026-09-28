import os

from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

from app.embedding import generate_embedding


load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

if not PINECONE_API_KEY:
    raise ValueError(
        "PINECONE_API_KEY is not configured. "
        "Add it to your .env file."
    )


pc = Pinecone(api_key=PINECONE_API_KEY)

INDEX_NAME = "pdf-embedding-index"
EMBEDDING_DIMENSION = 384


if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=EMBEDDING_DIMENSION,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-west-2"
        )
    )


index = pc.Index(INDEX_NAME)


def pdf_already_indexed(pdf_hash: str) -> bool:
    """
    Checks whether the first chunk of a PDF already exists
    in its dedicated Pinecone namespace.
    """
    first_chunk_id = f"{pdf_hash}_0"

    response = index.fetch(
        ids=[first_chunk_id],
        namespace=pdf_hash
    )

    return bool(response.vectors)


def index_pdf_chunks(chunks: list[str], pdf_hash: str) -> None:
    """
    Generates embeddings for PDF chunks and stores them
    inside a document-specific Pinecone namespace.
    """
    vectors = []

    for chunk_index, chunk in enumerate(chunks):
        embedding = generate_embedding(chunk)

        vectors.append(
            {
                "id": f"{pdf_hash}_{chunk_index}",
                "values": embedding.tolist(),
                "metadata": {
                    "text": chunk,
                    "chunk_index": chunk_index,
                    "document_id": pdf_hash
                }
            }
        )

    if vectors:
        index.upsert(
            vectors=vectors,
            namespace=pdf_hash
        )


def query_pinecone(
    query_embedding,
    pdf_hash: str,
    top_k: int = 5
):
    """
    Searches only within the namespace belonging
    to the requested PDF.
    """
    return index.query(
        vector=query_embedding.tolist(),
        top_k=top_k,
        include_metadata=True,
        namespace=pdf_hash
    )