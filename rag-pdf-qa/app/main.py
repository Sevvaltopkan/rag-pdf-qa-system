import requests
from fastapi import FastAPI
from pydantic import BaseModel
from app.processors.pdf_processor import split_pdf_to_chunks, split_text_to_chunks
from app.processors.embedding_generator import generate_embeddings
from app.database.database_manager import get_store_embeddings, get_query_embeddings

import logging

app = FastAPI()

logging.basicConfig(level=logging.INFO)

# API istekleri doğrulaması için model oluşturulur
class QueryRequest(BaseModel):
    url: str
    query: str

# FastAPI'de POST isteği olarak çalışan bir endpoint oluşturulur
@app.post("/process")
async def process_pdf_endpoint(request: QueryRequest):
    try:
        # PDF indirilir
        pdf_path = "downloaded_pdf.pdf"
        logging.info(f"Downloading PDF from {request.url}")
        download_pdf(request.url, pdf_path)

        # PDF parçalanır
        logging.info("Splitting PDF into chunks")
        chunks = split_pdf_to_chunks(pdf_path)
        if not chunks:
            logging.error("No text extracted from PDF")
            return {"error": "No text extracted from PDF"}

        # Embedding oluşturulur
        logging.info("Generating embeddings for chunks")
        embeddings = generate_embeddings(chunks)
        if not embeddings:
            logging.error("Failed to generate embeddings")
            return {"error": "Failed to generate embeddings"}

        # Embedding'leri veritabanına eklenir
        logging.info("Storing embeddings in the database")
        get_store_embeddings(embeddings)

        # Sorgu için embedding oluşturulur
        logging.info(f"Generating embedding for the query: {request.query}")
        query_chunks = split_text_to_chunks(request.query)
        query_embeddings = generate_embeddings(query_chunks)
        
        best_result = None
        for query_embedding in query_embeddings:
            results = get_query_embeddings(query_embedding)
            if not best_result or results[0]['score'] > best_result['score']:
                best_result = results[0]

        logging.info(f"Best query result: {best_result}")
        return {"results": best_result}

    except Exception as e:
        logging.error(f"An error occurred: {e}")
        return {"error": "Internal Server Error"}

# PDF indirilir
def download_pdf(url, output_path):
    response = requests.get(url)
    with open(output_path, 'wb') as file:
        file.write(response.content)
