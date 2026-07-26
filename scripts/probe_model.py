"""Probe whether an Ollama model loads and generates on this GPU."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request


def nvidia_mem():
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        ).strip()
        used, total, free = [x.strip() for x in out.split(",")]
        return {"used_mib": int(used), "total_mib": int(total), "free_mib": int(free)}
    except Exception as e:
        return {"error": str(e)}


def generate(model: str, prompt: str, timeout: int = 180) -> dict:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"num_predict": 32, "temperature": 0},
    }
    req = urllib.request.Request(
        "http://localhost:11434/api/generate",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode())
    return {
        "ok": True,
        "elapsed_s": round(time.time() - t0, 2),
        "response_preview": (data.get("response") or "")[:200],
        "eval_count": data.get("eval_count"),
        "eval_duration_ns": data.get("eval_duration"),
        "load_duration_ns": data.get("load_duration"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    args = ap.parse_args()
    model = args.model
    print(f"PROBE model={model}")
    print("mem_before", nvidia_mem())
    try:
        result = generate(model, "Reply with exactly: OK")
        print("generate", result)
        print("mem_after", nvidia_mem())
        print("RESULT=PASS")
        sys.exit(0)
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")[:500]
        print(f"HTTPError {e.code}: {body}")
        print("mem_after", nvidia_mem())
        print("RESULT=FAIL")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {type(e).__name__}: {e}")
        print("mem_after", nvidia_mem())
        print("RESULT=FAIL")
        sys.exit(1)


if __name__ == "__main__":
    main()
