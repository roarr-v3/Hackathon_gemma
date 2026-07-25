from app.schemas import ChatRequest


def test_chat_request_allows_plain_chat_without_documents():
    request = ChatRequest(
        conversation_id="plain-chat",
        message="Hello",
        document_ids=[],
    )

    assert request.document_ids == []
