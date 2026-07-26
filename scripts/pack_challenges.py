"""Pack a directory of ARC task JSON files into one challenges.json blob."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="Directory of per-task .json files")
    ap.add_argument("--out", required=True, help="Output challenges.json path")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    src = Path(args.dir)
    files = sorted(src.glob("*.json"))
    if args.limit:
        files = files[: args.limit]
    blob = {}
    for path in files:
        blob[path.stem] = json.loads(path.read_text(encoding="utf-8"))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(blob), encoding="utf-8")
    print(f"Wrote {out} with {len(blob)} tasks")


if __name__ == "__main__":
    main()
