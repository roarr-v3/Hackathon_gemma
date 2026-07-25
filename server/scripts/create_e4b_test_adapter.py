from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from safetensors.numpy import save_file


BASE_MODEL = "mlx-community/gemma-4-e4b-it-4bit"
LAYERS = 42
INPUT_SIZE = 10_240
OUTPUT_SIZE = 2_560


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a zero-effect PEFT LoRA used to verify E4B runtime routing."
    )
    parser.add_argument("output", type=Path)
    parser.add_argument("--rank", type=int, default=8)
    args = parser.parse_args()

    if args.rank <= 0:
        raise SystemExit("--rank must be positive")

    args.output.mkdir(parents=True, exist_ok=True)
    config = {
        "base_model_name_or_path": BASE_MODEL,
        "bias": "none",
        "inference_mode": True,
        "lora_alpha": args.rank,
        "lora_dropout": 0.0,
        "peft_type": "LORA",
        "r": args.rank,
        "target_modules": ["down_proj"],
        "task_type": "CAUSAL_LM",
    }
    (args.output / "adapter_config.json").write_text(
        json.dumps(config, indent=2) + "\n",
        encoding="utf-8",
    )

    tensors: dict[str, np.ndarray] = {}
    for layer in range(LAYERS):
        prefix = (
            "base_model.model.language_model.model.layers."
            f"{layer}.mlp.down_proj"
        )
        tensors[f"{prefix}.lora_A.weight"] = np.zeros(
            (args.rank, INPUT_SIZE), dtype=np.float16
        )
        tensors[f"{prefix}.lora_B.weight"] = np.zeros(
            (OUTPUT_SIZE, args.rank), dtype=np.float16
        )

    save_file(
        tensors,
        args.output / "adapter_model.safetensors",
        metadata={"format": "pt"},
    )
    print(
        f"Created rank-{args.rank} E4B test adapter with {LAYERS} layers "
        f"at {args.output}"
    )


if __name__ == "__main__":
    main()
