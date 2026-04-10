from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlparse

PROJECT_ROOT = Path(__file__).resolve().parent

DEFAULT_UPLOADS_DIR_NAME = "uploads"
DEFAULT_MAX_PDF_FILE_BYTES = 25 * 1024 * 1024
DEFAULT_MAX_QUESTION_CHARS = 1000
DEFAULT_MAX_CONTEXT_CHARS = 12000
DEFAULT_MAX_TOP_K = 20
DEFAULT_RATE_LIMIT_PER_MINUTE = 30
DEFAULT_RATE_LIMIT_WINDOW_SECONDS = 60
DEFAULT_RATE_LIMIT_DB_PATH = ".rate_limit.sqlite3"
DEFAULT_INNGEST_API_BASE = "http://127.0.0.1:8288/v1"
DEFAULT_QDRANT_URL = "http://127.0.0.1:6333"
DEFAULT_QDRANT_COLLECTION = "docs"
DEFAULT_INNGEST_APP_ID = "rag_app"
DEFAULT_EMBED_MODEL = "gemini-embedding-001"
DEFAULT_EMBED_DIM = 3072
DEFAULT_LLM_MODEL = "gemini-3-flash-preview"


def _get_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _get_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    try:
        return int(value.strip())
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be an integer.") from exc


def _get_str(name: str, default: str) -> str:
    value = os.getenv(name)
    if value is None:
        return default
    cleaned = value.strip()
    return cleaned or default


def _get_optional_str(name: str) -> str | None:
    value = os.getenv(name)
    if value is None:
        return None
    cleaned = value.strip()
    return cleaned or None


def _require_env(name: str) -> str:
    value = _get_optional_str(name)
    if value is None:
        raise ValueError(f"Missing required environment variable: {name}")
    return value


@dataclass(frozen=True)
class Settings:
    google_api_key: str
    inngest_app_id: str
    inngest_is_production: bool
    inngest_signing_key: str | None
    inngest_event_key: str | None
    inngest_api_base: str
    qdrant_url: str
    qdrant_api_key: str | None
    qdrant_collection: str
    embedding_model: str
    embedding_dim: int
    llm_model: str
    uploads_dir: Path
    rate_limit_db_path: Path
    max_pdf_file_bytes: int
    max_question_chars: int
    max_context_chars: int
    max_top_k: int
    rate_limit_per_minute: int
    rate_limit_window_seconds: int

    @property
    def is_local_dev(self) -> bool:
        return not self.inngest_is_production


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    uploads_dir_name = _get_str("UPLOADS_DIR", DEFAULT_UPLOADS_DIR_NAME)

    settings = Settings(
        google_api_key=_require_env("GOOGLE_API_KEY"),
        inngest_app_id=_get_str("INNGEST_APP_ID", DEFAULT_INNGEST_APP_ID),
        inngest_is_production=_get_bool("INNGEST_IS_PRODUCTION", False),
        inngest_signing_key=_get_optional_str("INNGEST_SIGNING_KEY"),
        inngest_event_key=_get_optional_str("INNGEST_EVENT_KEY"),
        inngest_api_base=_get_str("INNGEST_API_BASE", DEFAULT_INNGEST_API_BASE),
        qdrant_url=_get_str("QDRANT_URL", DEFAULT_QDRANT_URL),
        qdrant_api_key=_get_optional_str("QDRANT_API_KEY"),
        qdrant_collection=_get_str("QDRANT_COLLECTION", DEFAULT_QDRANT_COLLECTION),
        embedding_model=_get_str("EMBED_MODEL", DEFAULT_EMBED_MODEL),
        embedding_dim=_get_int("EMBED_DIM", DEFAULT_EMBED_DIM),
        llm_model=_get_str("LLM_MODEL", DEFAULT_LLM_MODEL),
        uploads_dir=_resolve_project_path(uploads_dir_name, "UPLOADS_DIR"),
        rate_limit_db_path=_resolve_project_path(
            _get_str("RATE_LIMIT_DB_PATH", DEFAULT_RATE_LIMIT_DB_PATH),
            "RATE_LIMIT_DB_PATH",
        ),
        max_pdf_file_bytes=_get_max_pdf_file_bytes(),
        max_question_chars=_get_int("MAX_QUESTION_CHARS", DEFAULT_MAX_QUESTION_CHARS),
        max_context_chars=_get_int("MAX_CONTEXT_CHARS", DEFAULT_MAX_CONTEXT_CHARS),
        max_top_k=_get_int("MAX_TOP_K", DEFAULT_MAX_TOP_K),
        rate_limit_per_minute=_get_int(
            "RATE_LIMIT_PER_MINUTE", DEFAULT_RATE_LIMIT_PER_MINUTE
        ),
        rate_limit_window_seconds=_get_int(
            "RATE_LIMIT_WINDOW_SECONDS", DEFAULT_RATE_LIMIT_WINDOW_SECONDS
        ),
    )

    validate_settings(settings)
    return settings


def _get_max_pdf_file_bytes() -> int:
    if _get_optional_str("MAX_PDF_FILE_BYTES") is not None:
        return _get_int("MAX_PDF_FILE_BYTES", DEFAULT_MAX_PDF_FILE_BYTES)

    max_pdf_mb = _get_optional_str("MAX_PDF_FILE_MB")
    if max_pdf_mb is not None:
        size_mb = _get_int(
            "MAX_PDF_FILE_MB", DEFAULT_MAX_PDF_FILE_BYTES // (1024 * 1024)
        )
        return size_mb * 1024 * 1024

    return DEFAULT_MAX_PDF_FILE_BYTES


def validate_settings(settings: Settings) -> None:
    if settings.inngest_is_production and not settings.inngest_signing_key:
        raise ValueError(
            "INNGEST_SIGNING_KEY is required when INNGEST_IS_PRODUCTION=true."
        )

    if settings.embedding_dim <= 0:
        raise ValueError("EMBED_DIM must be greater than 0.")

    if settings.max_pdf_file_bytes <= 0:
        raise ValueError("MAX_PDF_FILE_BYTES must be greater than 0.")

    if settings.max_question_chars <= 0:
        raise ValueError("MAX_QUESTION_CHARS must be greater than 0.")

    if settings.max_context_chars <= 0:
        raise ValueError("MAX_CONTEXT_CHARS must be greater than 0.")

    if settings.max_top_k <= 0:
        raise ValueError("MAX_TOP_K must be greater than 0.")

    if settings.rate_limit_per_minute <= 0:
        raise ValueError("RATE_LIMIT_PER_MINUTE must be greater than 0.")

    if settings.rate_limit_window_seconds <= 0:
        raise ValueError("RATE_LIMIT_WINDOW_SECONDS must be greater than 0.")

    _validate_http_url(settings.inngest_api_base, "INNGEST_API_BASE")
    _validate_localhost_url(settings.inngest_api_base, "INNGEST_API_BASE")
    _validate_http_url(settings.qdrant_url, "QDRANT_URL")


def _validate_http_url(value: str, env_name: str) -> None:
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"{env_name} must be a valid http(s) URL.")


def _validate_localhost_url(value: str, env_name: str) -> None:
    parsed = urlparse(value)
    host = (parsed.hostname or "").lower()
    if host not in {"127.0.0.1", "localhost"}:
        raise ValueError(f"{env_name} must target localhost or 127.0.0.1.")


def _resolve_project_path(value: str, env_name: str) -> Path:
    raw = Path(value.strip())
    candidate = raw if raw.is_absolute() else (PROJECT_ROOT / raw)
    resolved = candidate.resolve()
    project_root_resolved = PROJECT_ROOT.resolve()

    try:
        resolved.relative_to(project_root_resolved)
    except ValueError as exc:
        raise ValueError(f"{env_name} must stay within the project directory.") from exc

    return resolved


def sanitize_upload_filename(filename: str) -> str:
    name = Path(filename).name.strip()
    if not name:
        raise ValueError("Uploaded file must have a file name.")
    if name in {".", ".."}:
        raise ValueError("Invalid uploaded file name.")
    if not name.lower().endswith(".pdf"):
        raise ValueError("Only PDF files are allowed.")
    return name


def safe_upload_path(filename: str) -> Path:
    settings = get_settings()
    safe_name = sanitize_upload_filename(filename)
    path = (settings.uploads_dir / safe_name).resolve()

    try:
        path.relative_to(settings.uploads_dir)
    except ValueError as exc:
        raise ValueError("Unsafe upload path detected.") from exc

    return path


def ensure_runtime_ready() -> Settings:
    try:
        settings = get_settings()
    except ValueError as exc:
        sys.exit(str(exc))

    settings.uploads_dir.mkdir(parents=True, exist_ok=True)
    settings.rate_limit_db_path.parent.mkdir(parents=True, exist_ok=True)
    return settings
