from __future__ import annotations

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    server_name: str
    inference_ready: bool
    model: str


class CapabilitiesResponse(BaseModel):
    server_name: str
    base_model: str
    inference_ready: bool
    document_inference: bool = True
    supported_file_types: list[str]


class DocumentResponse(BaseModel):
    id: str
    name: str
    character_count: int


class ChatRequest(BaseModel):
    conversation_id: str
    message: str = Field(min_length=1, max_length=20_000)
    document_ids: list[str] = Field(min_length=1, max_length=20)
    adapter_id: str | None = None


class Citation(BaseModel):
    document_id: str
    document_name: str
    chunk_id: str


class ChatResponse(BaseModel):
    answer: str
    model: str
    citations: list[Citation]
