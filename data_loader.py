from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from llama_index.core.node_parser import SentenceSplitter
from llama_index.readers.file import PDFReader

from config import get_settings

load_dotenv()

_PDF_MAGIC = b"%PDF-"


@lru_cache(maxsize=1)
def get_gemini_client() -> genai.Client:
    settings = get_settings()
    return genai.Client(api_key=settings.google_api_key)


def _get_splitter() -> SentenceSplitter:
    return SentenceSplitter(chunk_size=1000, chunk_overlap=1)


def _validate_pdf_path(path: str | Path) -> Path:
    settings = get_settings()
    pdf_path = Path(path).resolve()

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file does not exist: {pdf_path}")

    if not pdf_path.is_file():
        raise ValueError(f"PDF path must point to a file: {pdf_path}")

    try:
        pdf_path.relative_to(settings.uploads_dir)
    except ValueError as exc:
        raise ValueError(
            f"PDF path must stay within the uploads directory: {pdf_path}"
        ) from exc

    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError("Only .pdf files are allowed.")

    if pdf_path.stat().st_size > settings.max_pdf_file_bytes:
        raise ValueError(
            f"PDF file exceeds the maximum allowed size of "
            f"{settings.max_pdf_file_bytes} bytes."
        )

    with pdf_path.open("rb") as handle:
        header = handle.read(len(_PDF_MAGIC))

    if header != _PDF_MAGIC:
        raise ValueError("Uploaded file is not a valid PDF.")

    return pdf_path


def load_and_chunk_pdf(path: str | Path) -> list[str]:
    pdf_path = _validate_pdf_path(path)
    docs = PDFReader().load_data(file=pdf_path)
    texts = [doc.text.strip() for doc in docs if getattr(doc, "text", None)]
    splitter = _get_splitter()

    chunks: list[str] = []
    for text in texts:
        if text:
            chunks.extend(
                chunk.strip() for chunk in splitter.split_text(text) if chunk.strip()
            )

    if not chunks:
        raise ValueError(f"No extractable text was found in PDF: {pdf_path.name}")

    return chunks


def embed_texts(texts: list[str]) -> list[list[float]]:
    cleaned_texts = [
        text.strip() for text in texts if isinstance(text, str) and text.strip()
    ]
    if not cleaned_texts:
        raise ValueError("At least one non-empty text input is required for embedding.")

    settings = get_settings()
    client = get_gemini_client()
    embeddings: list[list[float]] = []

    for text in cleaned_texts:
        response = client.models.embed_content(
            model=settings.embedding_model,
            contents=text,
        )

        if not response.embeddings:
            raise ValueError("Embedding response did not contain any embeddings.")

        values = response.embeddings[0].values
        if values is None:
            raise ValueError("Embedding response contained empty vector values.")

        vector = list(values)
        if len(vector) != settings.embedding_dim:
            raise ValueError(
                f"Unexpected embedding dimension: got {len(vector)}, "
                f"expected {settings.embedding_dim}."
            )

        embeddings.append(vector)

    return embeddings
