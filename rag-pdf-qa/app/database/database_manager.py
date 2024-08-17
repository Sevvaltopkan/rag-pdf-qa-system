
from pinecone import Pinecone, ServerlessSpec
import numpy as np

# API anahtarı entegresi
pc = Pinecone(
    api_key="e06ae660-7336-4c18-8eb8-979c9cb1df1a"
)
index_name = "my-index"

# İndeks mevcudiyet kontrolü
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1024,
        metric='cosine',
        spec=ServerlessSpec(
            cloud='aws',
            region='us-east1'
        )
    )

def get_store_embeddings(embeddings):
    index = pc.Index(index_name)
    for i, embedding in enumerate(embeddings):
        # Vektörü normalize et (L2 normu 1 olacak şekilde)
        norm = np.linalg.norm(embedding)
        if norm != 0:
            embedding = embedding / norm
        index.upsert([(str(i), embedding.tolist())])

def get_query_embeddings(query_embedding):
    index = pc.Index(index_name)
    query_vector = np.array(query_embedding, dtype=np.float32)
    
    # Vektörü normalize et (L2 normu 1 olacak şekilde)
    norm = np.linalg.norm(query_vector)
    if norm != 0:
        query_vector = query_vector / norm
    
    # Değerleri -1.0 ile 1.0 arasında olacak şekilde sınırlayın
    query_vector = np.clip(query_vector, -1.0, 1.0)
    
    # Pinecone'a uygun formatta (liste olarak) gönder
    response = index.query(queries=[query_vector.tolist()], top_k=5)
    return response['matches']