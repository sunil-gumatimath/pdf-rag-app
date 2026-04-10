from pathlib import Path

from dotenv import load_dotenv
from google import genai
from llama_index.core.node_parser import SentenceSplitter
from llama_index.readers.file import PDFReader

load_dotenv()

client = genai.Client()

EMBED_MODEL = "gemini-embedding-001"
EMBED_DIM = 3072

splitter = SentenceSplitter(chunk_size=1000, chunk_overlap=1)


def load_and_chunk_pdf(path: str) -> list[str]:
    docs = PDFReader().load_data(file=Path(path))
    texts = [d.text for d in docs if getattr(d, "text", None)]
    chunks = []
    for t in texts:
        chunks.extend(splitter.split_text(t))
    return chunks


def embed_texts(texts: list[str]) -> list[list[float]]:
    embeddings = []
    for text in texts:
        response = client.models.embed_content(
            model=EMBED_MODEL,
            contents=text,
        )
        if response.embeddings:
            values = response.embeddings[0].values
            if values is not None:
                embeddings.append(values)
    return embeddings
