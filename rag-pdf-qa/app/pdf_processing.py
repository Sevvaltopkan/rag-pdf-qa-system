from fastapi import HTTPException
import requests
from pdfminer.high_level import extract_text
import hashlib
from io import BytesIO
from app.embedding import generate_embedding
from app.pinecone_utils import index_pdf_chunks, pdf_already_indexed


# PDF dosyasını URL'den indirilmesi
def download_pdf(url):
    response = requests.get(url)
    if response.status_code == 200:
        return response.content
    else:
        raise HTTPException(status_code=404, detail="PDF not found at the provided URL.")

# Metni parçalara bölen fonksiyon tanımlanması
def chunk_pdf(text, chunk_size=100):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunks.append(' '.join(words[i:i + chunk_size]))
    return chunks

# PDF dosyasını indirip işlemeye ve Pinecone'da indekslenmesi
def process_and_index_pdf(url):
    pdf_content = download_pdf(url)
    if pdf_already_indexed(pdf_content):
        return "PDF already indexed."
    
    # BytesIO ile pdf_content'i işlenebilir bir formata dönüştürülmesi
    pdf_text = extract_text(BytesIO(pdf_content))
    chunks = chunk_pdf(pdf_text)
    pdf_hash = hashlib.md5(pdf_content).hexdigest()
    index_pdf_chunks(chunks, pdf_hash)
    return "PDF processed and indexed successfully."
