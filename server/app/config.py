from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _as_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    server_name: str
    api_token: str
    vllm_base_url: str
    vllm_api_key: str
    vllm_model: str
    data_dir: Path
    max_document_bytes: int
    mock_mode: bool

    @classmethod
    def from_environment(cls) -> "Settings":
        default_data_dir = Path(__file__).resolve().parents[1] / "data"
        return cls(
            server_name=os.getenv("GEMMA_SERVER_NAME", "Gemma Compute Node"),
            api_token=os.getenv("GEMMA_API_TOKEN", "gemma-demo"),
            vllm_base_url=os.getenv(
                "VLLM_BASE_URL",
                "http://127.0.0.1:8000/v1",
            ).rstrip("/"),
            vllm_api_key=os.getenv("VLLM_API_KEY", ""),
            vllm_model=os.getenv(
                "VLLM_MODEL",
                "mlx-community/gemma-4-e4b-it-4bit",
            ),
            data_dir=Path(os.getenv("GEMMA_DATA_DIR", str(default_data_dir))),
            max_document_bytes=int(
                os.getenv("GEMMA_MAX_DOCUMENT_BYTES", str(20 * 1024 * 1024))
            ),
            mock_mode=_as_bool(os.getenv("GEMMA_MOCK_MODE")),
        )
