"""Train the fixed local R1 rank-4 adapter, or audit training JSONL on CPU."""
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
    parser.add_argument("--output", required=True, type=Path, help="New run directory; existing paths are refused")
    parser.add_argument("--model", type=Path, default=ROOT / ".cache" / "deepseek-r1-distill-qwen-1.5b")
    parser.add_argument("--steps", type=int, required=True, help="Optimizer steps; four batch-1 microbatches per step")
    parser.add_argument("--target-supervised-tokens", type=int, help="Stop at first completed optimizer window reaching this budget; --steps is the hard cap")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    parser.add_argument("--dry-run", action="store_true", help="Tokenize, validate and hash only; no weights loaded or forward")
    args = parser.parse_args()
    from deltaweave.model import train_adapter
    result = train_adapter(args.data, args.model, args.output, steps=args.steps, seed=args.seed,
                           device=args.device, dry_run=args.dry_run,
                           target_supervised_tokens=args.target_supervised_tokens)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
