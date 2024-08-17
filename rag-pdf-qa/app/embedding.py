from transformers import AutoTokenizer, AutoModel
import torch


# Önceden eğitilmiş bir dil modeli ve tokenizer yüklenmesi
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")


# Verilen metin parçası  için embedding  oluşturulması
def generate_embedding(text_chunk):
    inputs = tokenizer(text_chunk, return_tensors="pt", truncation=True, padding=True)
    # Model kullanılarak embedding hesaplanıyor, burada gradient hesaplamaları devre dışı bırakılmıştır
    with torch.no_grad():
        embedding = model(**inputs).last_hidden_state.mean(dim=1).squeeze().numpy()
    return embedding
