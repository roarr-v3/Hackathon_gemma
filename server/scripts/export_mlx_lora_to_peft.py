from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert MLX-LM down_proj LoRA weights to vLLM PEFT format."
    )
    parser.add_argument("mlx_adapter", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--scale", type=float, default=1.0)
    args = parser.parse_args()

    source = args.mlx_adapter / "adapters.safetensors"
    tensors: dict[str, np.ndarray] = {}
    pairs: dict[str, dict[str, np.ndarray]] = {}
    with safe_open(source, framework="np") as weights:
        for key in weights.keys():
            if key.endswith(".lora_a"):
                pairs.setdefault(key.removesuffix(".lora_a"), {})["a"] = (
                    weights.get_tensor(key)
                )
            elif key.endswith(".lora_b"):
                pairs.setdefault(key.removesuffix(".lora_b"), {})["b"] = (
                    weights.get_tensor(key)
                )

    for module, pair in pairs.items():
        if set(pair) != {"a", "b"}:
            raise SystemExit(f"Incomplete MLX LoRA pair for {module}")
        if not module.endswith(".mlp.down_proj"):
            continue
        prefix = f"base_model.model.{module}"
        tensors[f"{prefix}.lora_A.weight"] = pair["a"].T.astype(
            np.float16, copy=False
        )
        tensors[f"{prefix}.lora_B.weight"] = pair["b"].T.astype(
            np.float16, copy=False
        )

    if not tensors:
        raise SystemExit("No down_proj LoRA tensors were found.")

    args.output.mkdir(parents=True, exist_ok=True)
    save_file(
        tensors,
        args.output / "adapter_model.safetensors",
        metadata={"format": "pt"},
    )
    config = {
        "base_model_name_or_path": "mlx-community/gemma-4-e4b-it-4bit",
        "bias": "none",
        "inference_mode": True,
        "lora_alpha": args.rank * args.scale,
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
    print(
        f"Exported {len(tensors) // 2} down_proj layers from "
        f"{source} to {args.output}"
    )


if __name__ == "__main__":
    main()
