# PDF RAG Application

A **Retrieval-Augmented Generation (RAG)** pipeline that ingests PDF documents, chunks and embeds them using Google Gemini, stores vectors in Qdrant, and exposes an async API via FastAPI + Inngest.

---

## Architecture

```
PDF File
   |
   v
PDFReader (LlamaIndex)
   |  extract text
   v
SentenceSplitter
   |  chunk text (1000 tokens, overlap 1)
   v
Google Gemini Embeddings (gemini-embedding-001, 3072-dim)
   |
   v
Qdrant Vector Store (localhost:6333)
   |
   v
FastAPI + Inngest (event-driven ingestion pipeline)
```

---

## Project Structure

```
pdf-rag-application/
├── main.py            # FastAPI app + Inngest event functions
├── data_loader.py     # PDF loading, chunking, and embedding
├── vector_db.py       # Qdrant vector store wrapper
├── custom_types.py    # Pydantic models for requests/responses
├── pyproject.toml     # Project dependencies (managed by uv)
├── .env               # Environment variables (not committed)
└── .gitignore
```

---

## Tech Stack

| Layer            | Tool                                                                 |
|------------------|----------------------------------------------------------------------|
| API Framework    | [FastAPI](https://fastapi.tiangolo.com/)                             |
| Event Queue      | [Inngest](https://www.inngest.com/)                                  |
| PDF Parsing      | [LlamaIndex PDFReader](https://docs.llamaindex.ai/)                  |
| Text Chunking    | LlamaIndex SentenceSplitter                                          |
| Embeddings       | [Google Gemini](https://ai.google.dev/) (`gemini-embedding-001`)     |
| Vector Store     | [Qdrant](https://qdrant.tech/)                                       |
| Data Validation  | [Pydantic](https://docs.pydantic.dev/)                               |
| Package Manager  | [uv](https://github.com/astral-sh/uv)                               |

---

## Getting Started

### Prerequisites

- Python 3.14+
- [uv](https://github.com/astral-sh/uv) package manager
- [Docker](https://www.docker.com/) (for Qdrant)
- A [Google AI API key](https://aistudio.google.com/app/apikey)

---

### 1. Clone the Repository

```bash
git clone https://github.com/sunil-gumatimath/pdf-rag-app.git
cd pdf-rag-app
```

### 2. Install Dependencies

```bash
uv sync
```

### 3. Set Up Environment Variables

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your_google_api_key_here
QDRANT_URL=http://localhost:6333
```

### 4. Start Qdrant

Run Qdrant locally using Docker:

```bash
# PowerShell
docker run -d --name qdrant -p 6333:6333 -v "${PWD}/qdrant_storage:/qdrant/storage" qdrant/qdrant

# CMD
docker run -d --name qdrant -p 6333:6333 -v "%cd%/qdrant_storage:/qdrant/storage" qdrant/qdrant
```

### 5. Start the Server

```bash
uv run uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`.

---

## API Usage

### Ingest a PDF

Send an Inngest event to trigger the ingestion pipeline:

```bash
curl -X POST http://localhost:8000/api/inngest \
  -H "Content-Type: application/json" \
  -d '{
    "name": "rag/ingest_pdf",
    "data": {
      "pdf_path": "path/to/your/document.pdf",
      "source_id": "my-document"
    }
  }'
```

**Event payload fields:**

| Field       | Type     | Required | Description                                          |
|-------------|----------|----------|------------------------------------------------------|
| `pdf_path`  | `string` | Yes      | Path to the PDF file on disk                         |
| `source_id` | `string` | No       | Identifier for the source (defaults to `pdf_path`)   |

---

## Pydantic Models

| Model            | Fields                                               | Description                            |
|------------------|------------------------------------------------------|----------------------------------------|
| `RAGChunkAndSrc` | `chunks: list[str]`, `source_id: str \| None`        | Chunked text with source reference     |
| `RAGUpsertResult`| `ingested: int`                                      | Number of chunks ingested              |
| `RAGSearchResult`| `contexts: list[str]`, `sources: list[str]`          | Retrieved context chunks and sources   |
| `RAGQueryResult` | `answer: str`, `sources: list[str]`, `num_contexts: int` | Final answer with metadata         |

---

## Ingestion Pipeline

The ingestion is handled as a two-step Inngest function (`RAG: Ingest PDF`):

1. **load_and_chunk** — Reads the PDF, extracts text, and splits it into chunks of 1000 tokens with an overlap of 1.
2. **embed_and_upsert** — Embeds each chunk using Gemini, generates deterministic UUIDs per chunk, and upserts the vectors into Qdrant.

---

## Vector Search

The `QdrantStorage.search()` method accepts a query vector and returns the top-k most similar chunks along with their source metadata:

```python
from vector_db import QdrantStorage

db = QdrantStorage()
results = db.search(query_vector=my_vector, top_k=5)
# { "contexts": [...], "sources": [...] }
```

---

## Development

### Run with Auto-reload

```bash
uv run uvicorn main:app --reload
```

### Activate Virtual Environment Manually

```bash
# Windows CMD
.venv\Scripts\activate

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

---

## Dependencies

```toml
fastapi >= 0.133.1
google-genai >= 1.14.0
inngest >= 0.5.16
llama-index-core >= 0.14.15
llama-index-readers-file >= 0.5.6
python-dotenv >= 1.2.1
qdrant-client >= 1.17.0
streamlit >= 1.54.0
uvicorn >= 0.41.0
```

---

## License

This project is for educational and personal use.