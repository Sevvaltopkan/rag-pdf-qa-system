# PDF Semantic Search & Retrieval API

A document-aware semantic retrieval system built with **Python, FastAPI, Hugging Face Transformers, Pinecone, and Docker**.

The application downloads a PDF from a URL, extracts and chunks its text, generates vector embeddings, stores them in a document-specific Pinecone namespace, and retrieves the most relevant passages for a natural-language question.

This project demonstrates the retrieval layer commonly used in **Retrieval-Augmented Generation (RAG)** systems and is designed to be extended with an LLM-based generation layer.

---

## Features

- PDF processing from public URLs
- Text extraction with `pdfminer.six`
- Overlapping text chunking for better context preservation
- Semantic embeddings using `sentence-transformers/all-MiniLM-L6-v2`
- Vector storage and similarity search with Pinecone
- Document isolation using Pinecone namespaces
- SHA-256 based document identification
- Duplicate document detection
- Configurable Top-K semantic retrieval
- Structured API responses with relevance scores and chunk metadata
- FastAPI automatic Swagger/OpenAPI documentation
- Health-check endpoint
- Environment-based secret management
- Docker support

---

## Architecture

```text
                    ┌───────────────────┐
PDF URL ───────────►│   PDF Downloader  │
                    └─────────┬─────────┘
                              │
                              ▼
                    ┌───────────────────┐
                    │  Text Extraction  │
                    │   (pdfminer.six)  │
                    └─────────┬─────────┘
                              │
                              ▼
                    ┌───────────────────┐
                    │ Chunk + Overlap   │
                    │  200 / 40 words   │
                    └─────────┬─────────┘
                              │
                              ▼
                    ┌───────────────────┐
                    │     Embedding     │
                    │  all-MiniLM-L6-v2│
                    └─────────┬─────────┘
                              │
                              ▼
                    ┌───────────────────┐
                    │     Pinecone      │
                    │   Vector Index    │
                    │ Namespace per PDF │
                    └─────────┬─────────┘
                              │
Natural Language Question ────┤
                              ▼
                    ┌───────────────────┐
                    │ Semantic Retrieval│
                    │   Cosine Search   │
                    └─────────┬─────────┘
                              │
                              ▼
                    Relevant PDF Chunks
                    + Scores + Metadata
```

Each PDF is identified using a deterministic SHA-256 hash. The hash is used as the document ID and Pinecone namespace, preventing retrieval results from different documents from being mixed.

---

## Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core application |
| FastAPI | REST API |
| Hugging Face Transformers | Embedding model |
| PyTorch | Model inference |
| all-MiniLM-L6-v2 | 384-dimensional text embeddings |
| Pinecone | Vector database and semantic search |
| pdfminer.six | PDF text extraction |
| Pydantic | Request/response validation |
| Docker | Containerization |
| python-dotenv | Environment configuration |

---

## Retrieval Pipeline

### 1. PDF Download

The API receives a public PDF URL and downloads the document with timeout and error handling.

### 2. Document Identification

A SHA-256 hash is generated from the PDF bytes:

```text
PDF bytes → SHA-256 → document_id
```

The same PDF therefore produces the same document identifier.

### 3. Text Extraction

Text is extracted from the PDF using `pdfminer.six`.

Documents without extractable text are rejected with an appropriate API error.

### 4. Chunking

Extracted text is divided into overlapping chunks.

Default configuration:

```text
Chunk size: 200 words
Overlap:     40 words
```

For example:

```text
Chunk 1 → words   1 - 200
Chunk 2 → words 161 - 360
Chunk 3 → words 321 - 520
```

The overlap helps preserve context that would otherwise be lost at chunk boundaries.

### 5. Embedding Generation

Each chunk is converted into a 384-dimensional vector using:

```text
sentence-transformers/all-MiniLM-L6-v2
```

### 6. Vector Storage

Embeddings are stored in Pinecone.

Each document receives its own namespace:

```text
Pinecone Index
│
├── namespace: document_A_hash
│   ├── chunk_0
│   ├── chunk_1
│   └── chunk_2
│
└── namespace: document_B_hash
    ├── chunk_0
    ├── chunk_1
    └── chunk_2
```

This isolates documents and prevents cross-document retrieval.

### 7. Semantic Retrieval

The user's question is embedded using the same model.

Pinecone performs cosine-similarity search only within the requested document's namespace and returns the most relevant chunks.

---

## API Endpoints

### Health Check

```http
GET /health
```

Response:

```json
{
  "status": "ok"
}
```

---

### Query a PDF

```http
POST /query/
```

Example request:

```json
{
  "url": "https://arxiv.org/pdf/2005.11401",
  "query": "What are the two types of memory combined in RAG?",
  "top_k": 5
}
```

Example response:

```json
{
  "query": "What are the two types of memory combined in RAG?",
  "document_id": "23e3249e9a1e...",
  "results": [
    {
      "text": "Relevant passage extracted from the PDF...",
      "score": 0.46,
      "chunk_index": 22
    }
  ]
}
```

`top_k` can be configured between `1` and `10`.

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Sevvaltopkan/rag-pdf-qa-system.git
cd rag-pdf-qa-system/rag-pdf-qa
```

### 2. Create a virtual environment

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Environment Configuration

Create a `.env` file in the repository root.

```env
PINECONE_API_KEY=your_pinecone_api_key
```

An `.env.example` file is included as a configuration template.

> Never commit API keys or other secrets to version control.

---

## Running the API

From the `rag-pdf-qa` directory:

```bash
uvicorn app.main:app --reload
```

The API will be available at:

```text
http://127.0.0.1:8000
```

Swagger documentation:

```text
http://127.0.0.1:8000/docs
```

---

## Docker

Build the image:

```bash
docker build -t pdf-semantic-search .
```

Run the container:

```bash
docker run \
  -p 8000:80 \
  -e PINECONE_API_KEY=your_pinecone_api_key \
  pdf-semantic-search
```

The API will then be available on:

```text
http://localhost:8000
```

---

## Engineering Decisions

### Why document-specific namespaces?

A global vector search can return chunks belonging to unrelated documents.

Using the SHA-256 document ID as the Pinecone namespace guarantees that a query is executed only against the requested PDF.

### Why SHA-256?

The document content itself determines its identifier.

This provides a deterministic document ID without relying on filenames or URLs, which may change even when the underlying document is identical.

### Why overlapping chunks?

Fixed non-overlapping chunks can split important context across boundaries.

A 40-word overlap helps preserve semantic continuity between neighboring chunks.

### Why separate retrieval metadata?

Each result includes:

- similarity score
- chunk index
- document identifier
- original chunk text

This makes retrieval behavior easier to inspect, debug, and extend.

---

## Error Handling

The API handles cases such as:

- unreachable PDF URLs
- HTTP download failures
- empty PDF responses
- PDFs without extractable text
- invalid `top_k` values
- missing Pinecone configuration

FastAPI and Pydantic provide additional request validation.

---

## Security

Secrets are loaded from environment variables instead of being stored directly in source code.

The repository includes:

```text
.env.example
```

while the real:

```text
.env
```

file is excluded through `.gitignore`.

---

## Current Scope

The current implementation focuses on the **retrieval component of a RAG architecture**:

```text
Document
   ↓
Chunking
   ↓
Embeddings
   ↓
Vector Database
   ↓
Semantic Retrieval
```

The API currently returns retrieved source passages rather than generating a synthesized answer.

A natural next step is adding an LLM generation layer:

```text
Question
   ↓
Semantic Retrieval
   ↓
Relevant Context
   ↓
LLM
   ↓
Grounded Answer
```

This separation keeps the retrieval layer independently testable and makes it possible to experiment with different LLM providers later.

---

## Future Improvements

- LLM-based answer generation
- Source citations in generated answers
- Token-aware chunking
- Batch embedding generation
- Async PDF processing
- Unit and integration tests
- Retrieval quality evaluation
- Hybrid keyword + vector search
- Support for local PDF uploads
- Configurable embedding models
- Request caching

---

## Project Status

The semantic retrieval pipeline has been tested end-to-end using public PDF documents through the FastAPI Swagger interface.

The current system successfully performs:

```text
PDF download
→ text extraction
→ document hashing
→ overlapping chunking
→ embedding generation
→ Pinecone indexing
→ document-specific semantic search
→ ranked retrieval
```

---

## Author

**Şevval Topkan**

Computer Engineer

GitHub: `Sevvaltopkan`