# PDF RAG Application

A Retrieval-Augmented Generation (RAG) application that ingests PDF documents, chunks and embeds them with Google Gemini, stores vectors in Qdrant, and exposes event-driven workflows through FastAPI + Inngest with a Streamlit UI.

## Overview

This project lets you:

- upload PDF files from a Streamlit interface
- trigger ingestion through Inngest
- extract and chunk PDF content with LlamaIndex
- generate embeddings with Google Gemini
- store vectors in Qdrant
- ask questions against the uploaded documents

## Architecture

```/dev/null/architecture.txt#L1-16
PDF upload
   |
   v
Streamlit UI
   |
   v
Inngest event
   |
   v
FastAPI + Inngest handler
   |
   v
PDFReader + SentenceSplitter
   |
   v
Google Gemini embeddings
   |
   v
Qdrant vector store
   |
   v
Gemini answer generation
```

## Project Structure

```/dev/null/project-structure.txt#L1-14
pdf-rag-application/
├── main.py
├── data_loader.py
├── vector_db.py
├── custom_types.py
├── config.py
├── streamlit_app.py
├── pyproject.toml
├── uv.lock
├── .env.example
├── .gitignore
├── README.md
├── qdrant_storage/
└── uploads/
```

## Tech Stack

- FastAPI
- Inngest
- Streamlit
- Google Gemini
- Qdrant
- LlamaIndex
- Pydantic
- python-dotenv
- uv

## Security Notes

This project includes several security-focused protections:

- environment-driven runtime configuration
- startup validation for required environment variables
- safer upload handling with filename sanitization
- upload path restriction to a project-controlled directory
- PDF header and file-size checks
- query length and retrieval-count limits
- localhost-only validation for the local Inngest API base
- request throttling with a persistent local SQLite-backed counter
- optional Qdrant API key support
- reduced prompt-injection risk through context limiting and stricter model instructions

Keep in mind:

- prompt injection in RAG systems can be reduced, but never fully eliminated
- local development defaults are not a substitute for production network security
- uploaded PDFs should still be treated as untrusted content

## Prerequisites

- Python 3.14+
- [uv](https://github.com/astral-sh/uv)
- [Docker](https://www.docker.com/)
- a Google AI API key
- optional: Inngest signing key for production
- optional: Qdrant API key if you secure your Qdrant instance

## Local Setup

### 1. Clone the repository

```/dev/null/setup.sh#L1-2
git clone https://github.com/sunil-gumatimath/pdf-rag-app.git
cd pdf-rag-app
```

### 2. Install dependencies

```/dev/null/setup.sh#L4-4
uv sync
```

### 3. Create `.env`

Copy the example file:

```/dev/null/env-copy.ps1#L1-2
Copy-Item .env.example .env
# or in CMD: copy .env.example .env
```

Then update it with your local values.

### 4. Minimum local `.env`

```/dev/null/example.env#L1-11
GOOGLE_API_KEY=your_google_ai_api_key_here
LLM_MODEL=gemini-3.1-pro-preview
EMBED_MODEL=gemini-embedding-001
QDRANT_URL=http://127.0.0.1:6333
QDRANT_COLLECTION=docs
INNGEST_APP_ID=rag_app
INNGEST_IS_PRODUCTION=false
INNGEST_API_BASE=http://127.0.0.1:8288/v1
MAX_PDF_FILE_BYTES=26214400
RATE_LIMIT_PER_MINUTE=30
UPLOADS_DIR=uploads
```

### 5. Start Qdrant

#### PowerShell

```/dev/null/qdrant-powershell.ps1#L1-7
docker run -d `
  --name qdrant `
  -p 6333:6333 `
  -v "${PWD}/qdrant_storage:/qdrant/storage" `
  qdrant/qdrant
```

#### CMD

```/dev/null/qdrant-cmd.bat#L1-1
docker run -d --name qdrant -p 6333:6333 -v "%cd%/qdrant_storage:/qdrant/storage" qdrant/qdrant
```

#### Optional: secure Qdrant with an API key

```/dev/null/qdrant-secure.ps1#L1-8
docker run -d `
  --name qdrant `
  -p 6333:6333 `
  -e QDRANT__SERVICE__API_KEY=your_qdrant_api_key `
  -v "${PWD}/qdrant_storage:/qdrant/storage" `
  qdrant/qdrant
```

If you enable a Qdrant API key, also set `QDRANT_API_KEY` in `.env`.

## Run Order

Start the app in this order.

### Terminal 1: FastAPI app

```/dev/null/run-api.sh#L1-1
uv run uvicorn main:app --reload
```

Runs on:

- `http://127.0.0.1:8000`

### Terminal 2: Inngest dev server

```/dev/null/run-inngest.sh#L1-1
npx inngest-cli@latest dev -u http://127.0.0.1:8000/api/inngest --no-discovery
```

### Terminal 3: Streamlit UI

```/dev/null/run-streamlit.sh#L1-1
uv run streamlit run .\streamlit_app.py
```

Runs on:

- `http://127.0.0.1:8501`

## Usage Flow

### Upload a PDF

The Streamlit UI will:

1. save the file into the configured uploads directory
2. validate the file name, extension, size, and PDF header
3. trigger an Inngest ingestion event
4. extract and chunk the document
5. embed chunks with Gemini
6. store vectors in Qdrant

### Ask a Question

The query flow will:

1. send a query event
2. validate question length and `top_k`
3. embed the user question
4. search matching chunks in Qdrant
5. send the retrieved context to Gemini
6. return an answer with sources

## Event Models

### Ingest Event

```/dev/null/ingest-event.json#L1-6
{
  "name": "rag/ingest_pdf",
  "data": {
    "pdf_path": "document.pdf",
    "source_id": "document.pdf"
  }
}
```

### Query Event

```/dev/null/query-event.json#L1-6
{
  "name": "rag/query_pdf_ai",
  "data": {
    "question": "Summarize the uploaded report",
    "top_k": 5
  }
}
```

## Configuration Reference

| Variable | Required | Description |
|---|---|---|
| `GOOGLE_API_KEY` | Yes | Google AI API key used for embeddings and generation |
| `INNGEST_APP_ID` | No | Inngest app identifier |
| `INNGEST_IS_PRODUCTION` | No | Enables production behavior for Inngest |
| `INNGEST_SIGNING_KEY` | Production only | Required when production mode is enabled |
| `INNGEST_EVENT_KEY` | No | Optional event key for hosted Inngest usage |
| `INNGEST_API_BASE` | No | Local Inngest API base for Streamlit polling |
| `QDRANT_URL` | No | Qdrant base URL |
| `QDRANT_API_KEY` | No | API key for secured Qdrant deployments |
| `QDRANT_COLLECTION` | No | Qdrant collection name |
| `EMBED_MODEL` | No | Embedding model name |
| `EMBED_DIM` | No | Embedding vector size |
| `LLM_MODEL` | No | Gemini model used for answer generation |
| `MAX_PDF_FILE_BYTES` | No | Maximum allowed upload size in bytes |
| `MAX_QUESTION_CHARS` | No | Maximum question length |
| `MAX_CONTEXT_CHARS` | No | Maximum context length passed to the model |
| `MAX_TOP_K` | No | Maximum retrieval count |
| `RATE_LIMIT_PER_MINUTE` | No | Request throttle limit per minute |
| `RATE_LIMIT_WINDOW_SECONDS` | No | Size of the throttle time window |
| `RATE_LIMIT_DB_PATH` | No | Path to the persistent throttle SQLite database |
| `UPLOADS_DIR` | No | Directory used for uploaded PDFs |

## Ignored Files and Local-Only Data

The following should stay out of version control:

- `.env`
- `.venv/`
- `uploads/`
- `qdrant_storage/`
- local throttle database files such as `.rate_limit.sqlite3`, `*.sqlite3`, and `*.db`
- IDE-specific folders like `.vscode/` and `.idea/`

Files you should commit:

- `uv.lock`
- `.env.example`
- source files
- `README.md`

## Production Notes

Before deploying to production:

- set `INNGEST_IS_PRODUCTION=true`
- set a valid `INNGEST_SIGNING_KEY`
- put the FastAPI app behind TLS
- restrict public access with network-level protections
- secure Qdrant with an API key if it is exposed outside localhost
- keep `.env` out of version control
- review upload size and throttle settings for your workload
- rotate API keys immediately if they are ever exposed

## Troubleshooting

### Missing environment variable errors

Make sure `.env` exists and includes at least:

```/dev/null/min-env.env#L1-2
GOOGLE_API_KEY=your_google_ai_api_key_here
QDRANT_URL=http://127.0.0.1:6333
```

### Streamlit not starting

Usually this means one of the following:

- `.env` is missing
- `GOOGLE_API_KEY` is missing
- `.env` was saved with the wrong variable name such as `GEMINI_API_KEY`
- the file is named `.env.txt` instead of `.env`

### Qdrant connection issues

- ensure Docker is running
- ensure Qdrant is listening on port `6333`
- verify `QDRANT_URL`
- if API key auth is enabled, verify `QDRANT_API_KEY`

### Inngest run output not appearing

- ensure the FastAPI app is running on port `8000`
- ensure the Inngest dev server is running
- ensure `INNGEST_API_BASE` points to `http://127.0.0.1:8288/v1`

### Upload issues

- confirm the uploaded file is a real PDF
- check the file size against `MAX_PDF_FILE_BYTES`
- ensure the uploads directory is writable

### Throttling behavior

- check `RATE_LIMIT_PER_MINUTE`
- verify the configured `RATE_LIMIT_DB_PATH` is writable
- remove the throttle database only when you intentionally want to reset local counters

## License

This project is intended for educational and personal use.