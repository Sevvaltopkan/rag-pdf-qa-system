from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.pdf_processing import process_and_index_pdf
from app.embedding import generate_embedding
from app.pinecone_utils import index, query_pinecone

app = FastAPI()

# Body içinden gelecek veriler için Pydantic modeli tanımlıyoruz
class QueryRequest(BaseModel):
    url: str
    query: str

@app.post("/query/")
def query_pdf(request: QueryRequest):
    # Gelen request'in body kısmından url ve query değerlerini alıyoruz
    url = request.url
    query = request.query
    
    process_and_index_pdf(url)
    
    query_embedding = generate_embedding(query)
    query_response = query_pinecone(query_embedding, top_k=5)
    
    results = [match['metadata']['text'] for match in query_response['matches']]
    return {"query": query, "results": results}
