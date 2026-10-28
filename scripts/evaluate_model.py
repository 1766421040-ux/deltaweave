"""Evaluate a pinned local base/adapter with raw receipts; never downloads a model."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="New aggregate JSON; per-case receipts are stored alongside")
    parser.add_argument("--model", type=Path, default=ROOT / ".cache" / "deepseek-r1-distill-qwen-1.5b")
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--max-seconds", type=float, default=120.0)
    parser.add_argument("--max-input-tokens", type=int, default=1024)
    parser.add_argument("--limit-dev", type=int, help="Diagnostic subset, only on dev; selected IDs are recorded")
    parser.add_argument("--resume", action="store_true", help="Reuse validated receipts of an unfinished run with identical settings")
    args = parser.parse_args()
    from deltaweave.model import evaluate_model
    result = evaluate_model(args.data, args.model, args.output, adapter_path=args.adapter, seed=args.seed,
                            device=args.device, max_new_tokens=args.max_new_tokens, max_seconds=args.max_seconds,
                            max_input_tokens=args.max_input_tokens, limit_dev=args.limit_dev, resume=args.resume)
    print(json.dumps({"status": result["status"], "metrics": result["metrics"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
