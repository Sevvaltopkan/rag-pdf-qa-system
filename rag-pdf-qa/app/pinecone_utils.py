import hashlib
from pinecone import Pinecone, ServerlessSpec

from app.embedding import generate_embedding

# Pinecone ayarlamaları
pc = Pinecone(api_key="e06ae660-7336-4c18-8eb8-979c9cb1df1a")

#Pinecone'da kullanılan indexin adı
index_name = "pdf-embedding-index"


#Belirtilen indexin oluşturulmaması durumunda yeni bir index oluşturulması
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,  # MiniLM modelinin çıktısı 384 boyutludur
        metric='cosine',
        spec=ServerlessSpec(
            cloud='aws',
            region='us-west-2'
        )
    )

#İndex kullanımı için referans alınması
index = pc.Index(index_name)

#PDF'in indekslenme kontrolünün yapılması
def pdf_already_indexed(pdf_content):
    pdf_text = pdf_content.decode('utf-8', errors='ignore')
    chunks = pdf_text.split()[:100]
    chunk_text = ' '.join(chunks)
    vector = generate_embedding(chunk_text)

    query_response = index.query(vector=vector.tolist(), top_k=1, include_metadata=True)
    return len(query_response['matches']) > 0

#PDF'in parçalara ayrılmasının ardından her bir parçanın Pinecone'a indekslenmesi
def index_pdf_chunks(chunks, pdf_hash):
    for i, chunk in enumerate(chunks):
        embedding = generate_embedding(chunk)
        # upsert() çağrısını doğru argümanlarla güncellenmesi
        index.upsert(vectors=[(f"{pdf_hash}_{i}", embedding, {"text": chunk})])

#Sorgu için Pinecone'da arama yapılması
def query_pinecone(query_embedding, top_k=5):
    return index.query(vector=query_embedding.tolist(), top_k=top_k, include_metadata=True)