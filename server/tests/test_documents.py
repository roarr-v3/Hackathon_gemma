from pathlib import Path

from app.documents import DocumentStore, chunk_text


def test_chunk_text_overlaps() -> None:
    words = [f"word-{index}" for index in range(20)]
    chunks = chunk_text(" ".join(words), size=10, overlap=2)

    assert len(chunks) == 3
    assert chunks[0].split()[-2:] == chunks[1].split()[:2]


def test_document_store_retrieves_matching_text(tmp_path: Path) -> None:
    store = DocumentStore(tmp_path)
    document = store.add(
        "notes.md",
        (
            b"The launch date is Friday. The project codename is Cactus. "
            b"The weather is not part of this document."
        ),
    )

    matches = store.retrieve([document.id], "What is the project codename?")

    assert matches
    assert matches[0].document_name == "notes.md"
    assert "Cactus" in matches[0].text
