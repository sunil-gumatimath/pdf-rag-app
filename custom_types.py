from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator

MAX_QUESTION_LENGTH = 1000
MAX_SOURCE_ID_LENGTH = 255
MIN_TOP_K = 1
MAX_TOP_K = 20


class RAGChunkAndSrc(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    chunks: list[str] = Field(default_factory=list)
    source_id: str | None = Field(default=None, max_length=MAX_SOURCE_ID_LENGTH)

    @field_validator("chunks")
    @classmethod
    def validate_chunks(cls, value: list[str]) -> list[str]:
        cleaned = [
            chunk.strip() for chunk in value if isinstance(chunk, str) and chunk.strip()
        ]
        if not cleaned:
            raise ValueError("At least one non-empty chunk is required.")
        return cleaned

    @field_validator("source_id")
    @classmethod
    def validate_source_id(cls, value: str | None) -> str | None:
        if value is None:
            return value
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("source_id cannot be empty when provided.")
        return cleaned


class RAGUpsertResult(BaseModel):
    ingested: int = Field(ge=0)


class RAGSearchResult(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    contexts: list[str] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)

    @field_validator("contexts")
    @classmethod
    def validate_contexts(cls, value: list[str]) -> list[str]:
        return [
            item.strip() for item in value if isinstance(item, str) and item.strip()
        ]

    @field_validator("sources")
    @classmethod
    def validate_sources(cls, value: list[str]) -> list[str]:
        return [
            item.strip() for item in value if isinstance(item, str) and item.strip()
        ]


class RAGQueryResult(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    answer: str = ""
    sources: list[str] = Field(default_factory=list)
    num_contexts: int = Field(ge=0)

    @field_validator("answer")
    @classmethod
    def validate_answer(cls, value: str) -> str:
        return value.strip()

    @field_validator("sources")
    @classmethod
    def validate_sources(cls, value: list[str]) -> list[str]:
        return [
            item.strip() for item in value if isinstance(item, str) and item.strip()
        ]


class IngestPDFEventData(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    pdf_path: str = Field(min_length=1, max_length=255)
    source_id: str | None = Field(default=None, max_length=MAX_SOURCE_ID_LENGTH)

    @field_validator("pdf_path")
    @classmethod
    def validate_pdf_path(cls, value: str) -> str:
        cleaned = value.strip().replace("\\", "/")
        if not cleaned:
            raise ValueError("pdf_path is required.")
        if "/" in cleaned:
            raise ValueError("pdf_path must be a file name only, not a path.")
        if cleaned.startswith("."):
            raise ValueError("pdf_path must not start with '.'.")
        if not cleaned.lower().endswith(".pdf"):
            raise ValueError("Only PDF files are allowed.")
        return cleaned

    @field_validator("source_id")
    @classmethod
    def validate_source_id(cls, value: str | None) -> str | None:
        if value is None:
            return value
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("source_id cannot be empty when provided.")
        return cleaned


class QueryPDFEventData(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    question: str = Field(min_length=1, max_length=MAX_QUESTION_LENGTH)
    top_k: int = Field(default=5, ge=MIN_TOP_K, le=MAX_TOP_K)

    @field_validator("question")
    @classmethod
    def validate_question(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("question is required.")
        return cleaned
