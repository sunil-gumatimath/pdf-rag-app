import logging
import uuid

import inngest
import inngest.fast_api
from dotenv import load_dotenv
from fastapi import FastAPI
from google import genai
from google.genai import types

from custom_types import (
    RAGChunkAndSrc,
    RAGQueryResult,
    RAGSearchResult,
    RAGUpsertResult,
)
from data_loader import embed_texts, load_and_chunk_pdf
from vector_db import QdrantStorage

load_dotenv()

inngest_client = inngest.Inngest(
    app_id="rag_app",
    logger=logging.getLogger("uvicorn"),
    is_production=False,
    serializer=inngest.PydanticSerializer(),
)

gemini_client = genai.Client()


@inngest_client.create_function(
    fn_id="RAG: Ingest PDF", trigger=inngest.TriggerEvent(event="rag/ingest_pdf")
)
async def rag_ingest_pdf(ctx: inngest.Context):
    async def _load(ctx: inngest.Context) -> RAGChunkAndSrc:
        pdf_path = ctx.event.data["pdf_path"]
        source_id = ctx.event.data.get("source_id", pdf_path)
        chunks = load_and_chunk_pdf(str(pdf_path))
        return RAGChunkAndSrc(chunks=chunks, source_id=str(source_id))

    async def _upsert(chunks_and_src: RAGChunkAndSrc) -> RAGUpsertResult:
        chunks = chunks_and_src.chunks
        source_id = chunks_and_src.source_id
        vecs = embed_texts(chunks)
        ids = [
            str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source_id}:{i}"))
            for i in range(len(chunks))
        ]
        payloads = [
            {"source": source_id, "text": chunks[i]} for i in range(len(chunks))
        ]
        QdrantStorage().upsert(ids, vecs, payloads)
        return RAGUpsertResult(ingested=len(chunks))

    chunks_and_src = await ctx.step.run(
        "load_and-chunk", lambda: _load(ctx), output_type=RAGChunkAndSrc
    )
    ingested = await ctx.step.run(
        "embed-and-upsert",
        lambda: _upsert(chunks_and_src),
        output_type=RAGUpsertResult,
    )
    return ingested.model_dump()


@inngest_client.create_function(
    fn_id="RAG: Query PDF", trigger=inngest.TriggerEvent(event="rag/query_pdf_ai")
)
async def rag_query_pdf_ai(ctx: inngest.Context):
    async def _search(question: str, top_k: int = 5) -> RAGSearchResult:
        query_vec = embed_texts([question])[0]
        store = QdrantStorage()
        found = store.search(query_vec, top_k)
        return RAGSearchResult(contexts=found["contexts"], sources=found["sources"])

    async def _generate(
        question: str, search_result: RAGSearchResult
    ) -> RAGQueryResult:
        context_block = "\n\n".join(f"- {c}" for c in search_result.contexts)
        user_content = (
            "Use the following context to answer the question.\n\n"
            f"Context:\n{context_block}\n\n"
            f"Question: {question}\n"
            "Answer concisely using the context above."
        )
        response = gemini_client.models.generate_content(
            model="gemini-2.0-flash",
            contents=user_content,
            config=types.GenerateContentConfig(
                system_instruction="Answer questions using only the provided context.",
                max_output_tokens=1024,
                temperature=0.2,
            ),
        )
        return RAGQueryResult(
            answer=response.text or "",
            sources=search_result.sources,
            num_contexts=len(search_result.contexts),
        )

    question = str(ctx.event.data["question"])
    top_k = int(str(ctx.event.data.get("top_k", 5)))

    found = await ctx.step.run(
        "embed-and-search",
        lambda: _search(question, top_k),
        output_type=RAGSearchResult,
    )

    result = await ctx.step.run(
        "llm-answer",
        lambda: _generate(question, found),
        output_type=RAGQueryResult,
    )

    return result.model_dump()


app = FastAPI()

inngest.fast_api.serve(app, inngest_client, [rag_ingest_pdf, rag_query_pdf_ai])
