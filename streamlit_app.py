import asyncio
import uuid
from pathlib import Path

import inngest
import requests
import streamlit as st
from dotenv import load_dotenv

from config import ensure_runtime_ready, safe_upload_path, sanitize_upload_filename
from custom_types import QueryPDFEventData

load_dotenv()
settings = ensure_runtime_ready()

REQUEST_TIMEOUT_SECONDS = 5.0
RUN_OUTPUT_TIMEOUT_SECONDS = 120.0
RUN_OUTPUT_POLL_INTERVAL_SECONDS = 0.5


st.set_page_config(page_title="RAG Ingest PDF", page_icon="📄", layout="centered")


@st.cache_resource
def get_inngest_client() -> inngest.Inngest:
    client_kwargs = {
        "app_id": settings.inngest_app_id,
        "is_production": settings.inngest_is_production,
    }
    if settings.inngest_event_key:
        client_kwargs["event_key"] = settings.inngest_event_key
    return inngest.Inngest(**client_kwargs)


def _is_pdf_bytes(file_bytes: bytes) -> bool:
    return file_bytes.startswith(b"%PDF-")


def save_uploaded_pdf(file) -> Path:
    safe_name = sanitize_upload_filename(file.name)
    file_bytes = bytes(file.getbuffer())

    if not file_bytes:
        raise ValueError("Uploaded file is empty.")

    if len(file_bytes) > settings.max_pdf_file_bytes:
        max_mb = settings.max_pdf_file_bytes // (1024 * 1024)
        raise ValueError(f"Uploaded file exceeds the {max_mb} MB limit.")

    if not _is_pdf_bytes(file_bytes):
        raise ValueError("Uploaded file is not a valid PDF.")

    unique_name = f"{uuid.uuid4().hex}_{safe_name}"
    file_path = safe_upload_path(unique_name)
    file_path.write_bytes(file_bytes)
    return file_path


async def send_rag_ingest_event(pdf_path: Path) -> None:
    client = get_inngest_client()
    await client.send(
        inngest.Event(
            name="rag/ingest_pdf",
            data={
                "pdf_path": pdf_path.name,
                "source_id": sanitize_upload_filename(pdf_path.name),
            },
        )
    )


st.title("Upload a PDF to Ingest")
uploaded = st.file_uploader("Choose a PDF", type=["pdf"], accept_multiple_files=False)

if uploaded is not None:
    try:
        with st.spinner("Uploading and triggering ingestion..."):
            path = save_uploaded_pdf(uploaded)
            asyncio.run(send_rag_ingest_event(path))
        st.success(
            f"Triggered ingestion for: {sanitize_upload_filename(uploaded.name)}"
        )
        st.caption("You can upload another PDF if you like.")
    except Exception as exc:
        st.error(str(exc))

st.divider()
st.title("Ask a question about your PDFs")


async def send_rag_query_event(question: str, top_k: int) -> str:
    payload = QueryPDFEventData(
        question=question[: settings.max_question_chars],
        top_k=min(top_k, settings.max_top_k),
    )

    client = get_inngest_client()
    result = await client.send(
        inngest.Event(
            name="rag/query_pdf_ai",
            data=payload.model_dump(),
        )
    )
    return str(result[0])


def _inngest_api_base() -> str:
    return settings.inngest_api_base.rstrip("/")


def fetch_runs(event_id: str) -> list[dict]:
    url = f"{_inngest_api_base()}/events/{event_id}/runs"
    response = requests.get(url, timeout=REQUEST_TIMEOUT_SECONDS)
    response.raise_for_status()
    data = response.json()
    runs = data.get("data", [])
    return runs if isinstance(runs, list) else []


def wait_for_run_output(
    event_id: str,
    timeout_s: float = RUN_OUTPUT_TIMEOUT_SECONDS,
    poll_interval_s: float = RUN_OUTPUT_POLL_INTERVAL_SECONDS,
) -> dict:
    import time

    start = time.time()
    last_status = None

    while True:
        runs = fetch_runs(event_id)
        if runs:
            run = runs[0]
            status = run.get("status")
            last_status = status or last_status

            if status in ("Completed", "Succeeded", "Success", "Finished"):
                output = run.get("output")
                return output if isinstance(output, dict) else {}

            if status in ("Failed", "Cancelled"):
                raise RuntimeError(f"Function run {status}")

        if time.time() - start > timeout_s:
            raise TimeoutError(
                f"Timed out waiting for run output (last status: {last_status})"
            )

        time.sleep(poll_interval_s)


with st.form("rag_query_form"):
    question = st.text_input(
        "Your question",
        max_chars=settings.max_question_chars,
        help=f"Maximum {settings.max_question_chars} characters.",
    )
    top_k = st.number_input(
        "How many chunks to retrieve",
        min_value=1,
        max_value=settings.max_top_k,
        value=min(5, settings.max_top_k),
        step=1,
    )
    submitted = st.form_submit_button("Ask")

    if submitted:
        try:
            payload = QueryPDFEventData(question=question, top_k=int(top_k))
            with st.spinner("Sending event and generating answer..."):
                event_id = asyncio.run(
                    send_rag_query_event(payload.question, payload.top_k)
                )
                output = wait_for_run_output(event_id)
                answer = str(output.get("answer", "")).strip()
                sources = output.get("sources", [])

            st.subheader("Answer")
            st.write(answer or "(No answer)")
            if isinstance(sources, list) and sources:
                st.caption("Sources")
                for source in sources:
                    st.write(f"- {source}")
        except Exception as exc:
            st.error(str(exc))
