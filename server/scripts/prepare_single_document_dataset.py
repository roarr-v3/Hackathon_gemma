from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create an MLX-LM raw-text dataset from exactly one UTF-8 file."
    )
    parser.add_argument("document", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    raw = args.document.read_bytes()
    text = raw.decode("utf-8").strip()
    if not text:
        raise SystemExit("The document is empty.")

    args.output.mkdir(parents=True, exist_ok=True)
    record = json.dumps({"text": text}, ensure_ascii=False) + "\n"
    (args.output / "train.jsonl").write_text(record, encoding="utf-8")
    (args.output / "valid.jsonl").write_text(record, encoding="utf-8")
    metadata = {
        "source_name": args.document.name,
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "source_bytes": len(raw),
        "training_records": 1,
    }
    (args.output / "source.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata))


if __name__ == "__main__":
    main()
