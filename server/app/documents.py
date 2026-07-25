from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass
from io import BytesIO
from pathlib import Path
from uuid import uuid4

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".markdown", ".json", ".csv"}
WORD_PATTERN = re.compile(r"[\w'-]+", re.UNICODE)


class DocumentError(ValueError):
    pass


@dataclass(frozen=True)
class StoredChunk:
    id: str
    text: str


@dataclass(frozen=True)
class StoredDocument:
    id: str
    name: str
    character_count: int
    chunks: list[StoredChunk]


@dataclass(frozen=True)
class RetrievedChunk:
    document_id: str
    document_name: str
    chunk_id: str
    text: str
    score: float


class DocumentStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def add(self, filename: str, data: bytes) -> StoredDocument:
        safe_name = Path(filename).name.strip()
        if not safe_name:
            raise DocumentError("The uploaded document has no filename.")

        extension = Path(safe_name).suffix.lower()
        if extension not in SUPPORTED_EXTENSIONS:
            supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
            raise DocumentError(f"Unsupported file type. Use one of: {supported}.")

        text = extract_text(extension, data).strip()
        if not text:
            raise DocumentError("No readable text was found in the document.")

        document_id = str(uuid4())
        chunks = [
            StoredChunk(id=f"{document_id}:{index}", text=chunk)
            for index, chunk in enumerate(chunk_text(text))
        ]
        document = StoredDocument(
            id=document_id,
            name=safe_name,
            character_count=len(text),
            chunks=chunks,
        )
        path = self.root / f"{document_id}.json"
        path.write_text(json.dumps(asdict(document), ensure_ascii=False), encoding="utf-8")
        return document

    def get(self, document_id: str) -> StoredDocument:
        if not re.fullmatch(r"[0-9a-f-]{36}", document_id):
            raise DocumentError("Invalid document identifier.")
        path = self.root / f"{document_id}.json"
        if not path.is_file():
            raise DocumentError(f"Document {document_id} was not found.")
        payload = json.loads(path.read_text(encoding="utf-8"))
        return StoredDocument(
            id=payload["id"],
            name=payload["name"],
            character_count=payload["character_count"],
            chunks=[StoredChunk(**chunk) for chunk in payload["chunks"]],
        )

    def retrieve(
        self,
        document_ids: list[str],
        query: str,
        limit: int = 5,
    ) -> list[RetrievedChunk]:
        query_terms = set(tokenize(query))
        candidates: list[RetrievedChunk] = []

        for document_id in dict.fromkeys(document_ids):
            document = self.get(document_id)
            for chunk in document.chunks:
                chunk_terms = tokenize(chunk.text)
                if query_terms:
                    overlap = sum(1 for term in chunk_terms if term in query_terms)
                    unique_overlap = len(query_terms.intersection(chunk_terms))
                    score = (overlap + unique_overlap * 2) / math.sqrt(
                        max(len(chunk_terms), 1)
                    )
                else:
                    score = 0
                candidates.append(
                    RetrievedChunk(
                        document_id=document.id,
                        document_name=document.name,
                        chunk_id=chunk.id,
                        text=chunk.text,
                        score=score,
                    )
                )

        candidates.sort(key=lambda chunk: chunk.score, reverse=True)
        return candidates[:limit]


def extract_text(extension: str, data: bytes) -> str:
    if extension == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as error:
            raise DocumentError(
                "PDF support requires the pypdf server dependency."
            ) from error
        reader = PdfReader(BytesIO(data))
        return "\n\n".join(page.extract_text() or "" for page in reader.pages)
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as error:
        raise DocumentError("The document is not valid UTF-8 text.") from error


def chunk_text(text: str, size: int = 260, overlap: int = 50) -> list[str]:
    words = text.split()
    if not words:
        return []
    if overlap >= size:
        raise ValueError("Chunk overlap must be smaller than chunk size.")

    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = min(start + size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start = end - overlap
    return chunks


def tokenize(text: str) -> list[str]:
    return [match.group(0).lower() for match in WORD_PATTERN.finditer(text)]
