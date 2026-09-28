import hashlib
from io import BytesIO

import requests
from fastapi import HTTPException
from pdfminer.high_level import extract_text

from app.pinecone_utils import index_pdf_chunks, pdf_already_indexed


DEFAULT_CHUNK_SIZE = 200
DEFAULT_CHUNK_OVERLAP = 40
DOWNLOAD_TIMEOUT = 20


def download_pdf(url: str) -> bytes:
    """
    Downloads a PDF from the provided URL.
    """
    try:
        response = requests.get(url, timeout=DOWNLOAD_TIMEOUT)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise HTTPException(
            status_code=400,
            detail="Unable to download the PDF from the provided URL."
        ) from exc

    if not response.content:
        raise HTTPException(
            status_code=400,
            detail="The downloaded PDF is empty."
        )

    return response.content


def generate_document_id(pdf_content: bytes) -> str:
    """
    Generates a deterministic SHA-256 identifier for a PDF.

    The same PDF content will always produce the same document ID.
    """
    return hashlib.sha256(pdf_content).hexdigest()


def extract_pdf_text(pdf_content: bytes) -> str:
    """
    Extracts text from PDF bytes.
    """
    try:
        text = extract_text(BytesIO(pdf_content))
    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail="The PDF could not be processed."
        ) from exc

    text = text.strip()

    if not text:
        raise HTTPException(
            status_code=422,
            detail="No extractable text was found in the PDF."
        )

    return text


def chunk_pdf(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP
) -> list[str]:
    """
    Splits text into overlapping word-based chunks.

    Overlap preserves some context between neighboring chunks.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            "overlap must be greater than or equal to zero "
            "and smaller than chunk_size."
        )

    words = text.split()

    if not words:
        return []

    chunks = []
    step = chunk_size - overlap

    for start in range(0, len(words), step):
        chunk_words = words[start:start + chunk_size]

        if not chunk_words:
            break

        chunks.append(" ".join(chunk_words))

        if start + chunk_size >= len(words):
            break

    return chunks


def process_and_index_pdf(url: str) -> str:
    """
    Downloads, identifies, extracts and indexes a PDF.

    Returns the document ID used as the Pinecone namespace.
    """
    pdf_content = download_pdf(url)

    document_id = generate_document_id(pdf_content)

    if pdf_already_indexed(document_id):
        return document_id

    pdf_text = extract_pdf_text(pdf_content)
    chunks = chunk_pdf(pdf_text)

    if not chunks:
        raise HTTPException(
            status_code=422,
            detail="The PDF did not produce any indexable text chunks."
        )

    index_pdf_chunks(
        chunks=chunks,
        pdf_hash=document_id
    )

    return document_id