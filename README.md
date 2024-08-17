# RAG Tabanlı Soru-Cevap Sistemi

Bu proje, PDF dokümanlarından bilgi çıkarımı yapabilen ve sorulara cevap verebilen bir RESTful API servisidir. Proje, FastAPI, Pinecone, transformers ve pdfminer.six gibi kütüphaneler kullanılarak geliştirilmiştir.

## Kurulum

### Gerekli Bağımlılıkları Yükleyin

Öncelikle, projenin bağımlılıklarını yüklemek için aşağıdaki komutu kullanın:

```bash
pip install -r requirements.txt
```

### Pinecone API Anahtarı

Projenin çalışabilmesi için Pinecone API anahtarına ihtiyacınız var. API anahtarınızı çevresel bir değişken olarak ayarlayın veya kod içinde doğrudan kullanın.

```bash
pc = Pinecone(api_key="YOUR_PINECONE_API_KEY")
```

## Kullanım

Proje, bir PDF dosyasını indirir, içeriğini işler ve sorgulama yapmanızı sağlar. API, URL ve sorguyu JSON formatında alır ve en alakalı sonuçları döner.

### API Endpoint
- POST /query/: Bu endpoint, bir PDF dosyasını indirir, içeriklerini işler ve verilen sorgu ile en alakalı sonuçları döner.

JSON Body
- `url`: PDF dosyasının URL'si (string)
- `query`: Sorgulamak istediğiniz metin (string)

Örnek JSON
```json
{
  "url": "https://example.com/sample.pdf",
  "query": "What is the meaning of life?"
}
```

Örnek cURL Komutu

```bash
curl -X POST "http://localhost:8000/query/" -H "Content-Type: application/json" -d '{"url":"https://example.com/sample.pdf", "query":"What is the meaning of life?"}'
```

## Projeyi Çalıştırma

Aşağıdaki komutla FastAPI uygulamasını başlatabilirsiniz:

```bash
uvicorn app.main:app --reload
```

```css
rag-pdf-qa-system/
│
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── pdf_processing.py
│   ├── embedding.py
│   └── pinecone_utils.py
│
├── Dockerfile
├── requirements.txt
└── README.md
```

### Docker ile Çalıştırma

Projenizi Docker kullanarak çalıştırmak için aşağıdaki adımları izleyin:

1. Docker image oluşturun:
```bash
docker build -t pdf-qa-api .
```

2. Docker container'ı çalıştırın:
```bash
docker run -d -p 80:80 pdf-qa-api
```