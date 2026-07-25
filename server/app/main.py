from __future__ import annotations

import hmac
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile, status

from .config import Settings
from .documents import DocumentError, DocumentStore, RetrievedChunk
from .schemas import (
    CapabilitiesResponse,
    ChatRequest,
    ChatResponse,
    Citation,
    DocumentResponse,
    HealthResponse,
)
from .vllm_client import InferenceBackendError, VLLMClient


settings = Settings.from_environment()
document_store = DocumentStore(settings.data_dir / "documents")
backend = VLLMClient(
    base_url=settings.vllm_base_url,
    api_key=settings.vllm_api_key,
    default_model=settings.vllm_model,
    mock_mode=settings.mock_mode,
)


@asynccontextmanager
async def lifespan(_: FastAPI):
    yield
    await backend.close()


app = FastAPI(
    title="Gemma Personal Compute Mesh",
    version="0.1.0",
    lifespan=lifespan,
)


def require_api_token(
    x_api_key: Annotated[str | None, Header()] = None,
) -> None:
    if not settings.api_token:
        return
    if x_api_key is None or not hmac.compare_digest(x_api_key, settings.api_token):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid compute-node API token.",
        )


Authenticated = Annotated[None, Depends(require_api_token)]


@app.get("/v1/health", response_model=HealthResponse)
async def health(_: Authenticated) -> HealthResponse:
    ready = await backend.health()
    return HealthResponse(
        status="ok" if ready else "degraded",
        server_name=settings.server_name,
        inference_ready=ready,
        model=settings.vllm_model,
    )


@app.get("/v1/capabilities", response_model=CapabilitiesResponse)
async def capabilities(_: Authenticated) -> CapabilitiesResponse:
    return CapabilitiesResponse(
        server_name=settings.server_name,
        base_model=settings.vllm_model,
        inference_ready=await backend.health(),
        supported_file_types=["pdf", "txt", "md", "markdown", "json", "csv"],
    )


@app.post("/v1/documents", response_model=DocumentResponse)
async def upload_document(
    _: Authenticated,
    file: Annotated[UploadFile, File()],
) -> DocumentResponse:
    data = await file.read(settings.max_document_bytes + 1)
    if len(data) > settings.max_document_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="The document exceeds the configured size limit.",
        )
    try:
        document = document_store.add(file.filename or "document", data)
    except DocumentError as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(error),
        ) from error
    return DocumentResponse(
        id=document.id,
        name=document.name,
        character_count=document.character_count,
    )


@app.post("/v1/chat/completions", response_model=ChatResponse)
async def chat(
    _: Authenticated,
    request: ChatRequest,
) -> ChatResponse:
    try:
        retrieved = document_store.retrieve(
            request.document_ids,
            request.message,
            limit=5,
        )
    except DocumentError as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(error),
        ) from error

    context = format_context(retrieved)
    messages = [
        {
            "role": "system",
            "content": (
                "You are Gemma running on the user's private compute node. "
                "Answer using the supplied document context. If the context does "
                "not contain the answer, say that clearly. Cite sources inline as "
                "[Source 1], [Source 2], and so on. Do not invent citations."
            ),
        },
        {
            "role": "user",
            "content": (
                f"DOCUMENT CONTEXT\n{context}\n\n"
                f"USER QUESTION\n{request.message}"
            ),
        },
    ]

    selected_model = request.adapter_id or settings.vllm_model
    try:
        answer, used_model = await backend.chat(messages, model=selected_model)
    except InferenceBackendError as error:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(error),
        ) from error

    citations = [
        Citation(
            document_id=chunk.document_id,
            document_name=chunk.document_name,
            chunk_id=chunk.chunk_id,
        )
        for chunk in retrieved
    ]
    return ChatResponse(answer=answer, model=used_model, citations=citations)


def format_context(chunks: list[RetrievedChunk]) -> str:
    if not chunks:
        return "No relevant document context was retrieved."
    sections = []
    for index, chunk in enumerate(chunks, start=1):
        sections.append(
            f"[Source {index}: {chunk.document_name}]\n{chunk.text}"
        )
    return "\n\n".join(sections)
