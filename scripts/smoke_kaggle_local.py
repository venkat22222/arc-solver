"""Smoke-test kaggle_local: load Qwen3-8B (4-bit) and generate one short reply.

Expect this to be slow on a 4GB laptop GPU (CPU offload). Goal is correctness,
not speed — catch import / device_map / generate bugs before Kaggle.

Usage:
  pip install -r requirements.txt
  # CUDA torch (example):
  #   pip install torch --index-url https://download.pytorch.org/whl/cu124
  python -m scripts.smoke_kaggle_local
  python -m scripts.smoke_kaggle_local --model Qwen/Qwen3-8B --max-tokens 32
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.llm_client import LLMClient, load_config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--config", default=None, help="Optional config.yaml path")
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--prompt", default="Reply with exactly: PONG")
    ap.add_argument("--no-4bit", action="store_true", help="Disable bitsandbytes (needs lots of RAM)")
    args = ap.parse_args()

    cfg = load_config(args.config) if args.config else {}
    kl = dict(cfg.get("kaggle_local") or {})
    if args.no_4bit:
        kl["load_in_4bit"] = False
        kl.setdefault("torch_dtype", "float16")

    # Ensure laptop offload defaults if not provided
    kl.setdefault("load_in_4bit", True)
    kl.setdefault("device_map", "auto")
    if "max_memory" not in kl and kl.get("load_in_4bit"):
        kl["max_memory"] = {0: "3200MB", "cpu": "24GB"}

    print("=== smoke_kaggle_local ===")
    try:
        import torch

        print(f"torch={torch.__version__} cuda={torch.cuda.is_available()} "
              f"cuda_ver={torch.version.cuda}")
        if torch.cuda.is_available():
            print(f"gpu={torch.cuda.get_device_name(0)} "
                  f"vram={torch.cuda.get_device_properties(0).total_memory/1e9:.2f}GB")
    except ImportError as e:
        print("FAIL: torch not installed:", e)
        sys.exit(1)

    t0 = time.time()
    client = LLMClient(backend="kaggle_local", model_name=args.model, **kl)
    load_s = time.time() - t0
    print(f"load_seconds={load_s:.1f}")

    t1 = time.time()
    out = client.generate(args.prompt, max_tokens=args.max_tokens, temperature=0.0)
    gen_s = time.time() - t1
    print(f"gen_seconds={gen_s:.1f}")
    print("OUTPUT:", repr(out[:500]))
    if not out.strip():
        print("FAIL: empty generation")
        sys.exit(2)
    print("OK: kaggle_local load+generate succeeded")


if __name__ == "__main__":
    main()
