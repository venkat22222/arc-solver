"""Smoke-test kaggle_local: load a Qwen3 model (4-bit) and generate one short reply.

On a 4GB laptop GPU, prefer Qwen3-4B for a full generate smoke (same client code).
Qwen3-8B NF4 needs ~5GB+ VRAM; CPU offload of 4-bit layers is currently broken on
Windows bitsandbytes (uint8 re-quantize). Use config.kaggle.yaml on P100 for 8B.

Usage:
  python -m scripts.smoke_kaggle_local --model Qwen/Qwen3-4B
  python -m scripts.smoke_kaggle_local --model Qwen/Qwen3-8B
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
    ap.add_argument("--no-4bit", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config) if args.config else load_config()
    kl = dict(cfg.get("kaggle_local") or {})
    if args.no_4bit:
        kl["load_in_4bit"] = False
        kl.setdefault("torch_dtype", "float16")

    kl.setdefault("load_in_4bit", True)
    kl.setdefault("device_map", "auto")
    kl.setdefault("llm_int8_enable_fp32_cpu_offload", False)

    print("=== smoke_kaggle_local ===")
    try:
        import torch

        print(
            f"torch={torch.__version__} cuda={torch.cuda.is_available()} "
            f"cuda_ver={torch.version.cuda}"
        )
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            print(
                f"gpu={torch.cuda.get_device_name(0)} "
                f"vram={props.total_memory / 1e9:.2f}GB"
            )
    except ImportError as e:
        print("FAIL: torch not installed:", e)
        sys.exit(1)

    t0 = time.time()
    try:
        client = LLMClient(backend="kaggle_local", model_name=args.model, **kl)
    except ValueError as e:
        msg = str(e)
        if "CPU or the disk" in msg or "enough GPU RAM" in msg:
            print("FAIL: model does not fit in GPU VRAM without 4-bit CPU offload.")
            print(
                "On Windows, bitsandbytes CPU-offload of 4-bit weights breaks at "
                "generate-time (uint8 re-quantize). Options:"
            )
            print("  1) Smoke with a smaller twin: --model Qwen/Qwen3-4B")
            print("  2) Run Qwen3-8B on Kaggle P100 via config.kaggle.yaml")
            print("Detail:", msg[:300])
            sys.exit(3)
        raise
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
