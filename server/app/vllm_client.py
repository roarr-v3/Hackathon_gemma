from __future__ import annotations

from typing import Any

import httpx


class InferenceBackendError(RuntimeError):
    pass


class VLLMClient:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        default_model: str,
        mock_mode: bool = False,
    ) -> None:
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.client = httpx.AsyncClient(
            base_url=base_url,
            headers=headers,
            timeout=httpx.Timeout(180.0, connect=5.0),
        )
        self.default_model = default_model
        self.mock_mode = mock_mode

    async def close(self) -> None:
        await self.client.aclose()

    async def health(self) -> bool:
        if self.mock_mode:
            return True
        try:
            response = await self.client.get("models")
            response.raise_for_status()
            return True
        except httpx.HTTPError:
            return False

    async def chat(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
    ) -> tuple[str, str]:
        selected_model = model or self.default_model
        if self.mock_mode:
            return (
                "Mock compute-node response. The document reached the Mac and the "
                "routing path is working; start vLLM for a real Gemma answer.",
                selected_model,
            )

        payload: dict[str, Any] = {
            "model": selected_model,
            "messages": messages,
            "temperature": 0.2,
            "max_tokens": 700,
            "stream": False,
        }
        try:
            response = await self.client.post("chat/completions", json=payload)
            response.raise_for_status()
            body = response.json()
            answer = body["choices"][0]["message"]["content"]
        except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError) as error:
            raise InferenceBackendError(f"vLLM request failed: {error}") from error

        if not isinstance(answer, str) or not answer.strip():
            raise InferenceBackendError("vLLM returned an empty answer.")
        return answer.strip(), selected_model
