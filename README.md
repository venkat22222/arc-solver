# ARC-AGI-2 Solver

LLM-driven generate–execute–verify pipeline for ARC-AGI-2 puzzles. **No fine-tuning** — prompting + orchestration only, with a config-swappable model backend for local (Ollama / API) and Kaggle (in-process HF model).

## Setup

```bash
cd arc-solver
pip install -r requirements.txt
```

### CUDA + kaggle_local (Qwen3-8B 4-bit)

`kaggle_local` needs a **CUDA** build of PyTorch (the default `pip install torch` CPU wheel will not work with bitsandbytes):

```bash
# Example — pick the CUDA index that matches your driver (cu124 is a common choice)
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

Smoke-test load + generate (slow on a 4GB laptop; uses CPU offload via `config.yaml` `max_memory`):

```bash
python -m scripts.smoke_kaggle_local --model Qwen/Qwen3-8B --max-tokens 32
```

Kaggle target config: `config.kaggle.yaml` (P100, no CPU offload). Notebook: `notebooks/kaggle_submission.ipynb`.

For local smoke tests with Ollama:

```bash
ollama pull qwen2.5:1.5b
# ensure ollama serve is running on localhost:11434
```

Edit `config.yaml` to switch backends:

```yaml
backend: "ollama"          # laptop dev
# backend: "api"           # hosted API (dev only)
# backend: "kaggle_local"  # in-process HF 4-bit (see config.kaggle.yaml)
```

## Data

Sample training puzzles live in `data/arc-agi-2/training/`. Full dataset from the official source:

- https://github.com/arcprize/ARC-AGI-2

```bash
# clone or copy training/ + evaluation/ JSON files into data/arc-agi-2/
```

## Pipeline stages

0. **Constraints** (`constraints.py`) — hard size/color/object/bg rules from train pairs  
1a. **Abstractions** — LLM proposes multiple plain-English hypotheses  
1b. **Code gen** — each hypothesis → `solve(grid)` using library primitives  
2. **Sandbox + self-debug** — safe exec + trace feedback retries  
3. **Holistic judge** — compare full reasoning traces (not majority vote on grids)  
4. **MDL tiebreak** — prefer simpler code / fewer magic numbers  
(+ early stop when zero progress after N cycles)

## Tests (no LLM)

```bash
cd arc-solver
python -m pytest tests/ -v
```

Smoke-print preprocess + constraints on sample puzzles:

```bash
python -m scripts.smoke_preprocess
```

## Build status (Section 8)

| Step | Module | Status |
|------|--------|--------|
| 1 | `loader.py` + sample data | done — full set: **1000 train / 120 eval** |
| 2 | `preprocess.py` | done |
| 3 | `constraints.py` | done |
| 4 | `sandbox.py` | done |
| 5 | `library.py` | done |
| 6 | `llm_client.py` | done — `ollama` + `mock` + `api` + `kaggle_local` |
| 7 | prompt templates | done |
| 8 | `self_debug.py` | done |
| 9–10 | `pipeline.py` + easy puzzles | **mock E2E 5/5**; Ollama 1.5B smoke runs but too weak to solve |
| 11 | `judge.py` + `tiebreak.py` | done |
| 12 | `early_stop.py` | done |
| 13–14 | Kaggle notebook + full-set timing | notebook + `build_submission` wired; run on P100 next |

### Quick commands

```bash
# Pipeline wiring (no model) — should hit 5/5 on easy geometry set
python -m scripts.run_e2e --backend mock --all-easy

# Local Ollama (after: ollama pull qwen2.5:1.5b)
python -m scripts.run_e2e --backend ollama --puzzle 6150a2bd --budget 300

# Pack evaluation dir → challenges blob, then dry-run submission (limit N)
python -m scripts.pack_challenges --dir data/arc-agi-2/evaluation --out data/arc-agi-2/evaluation_challenges.json
python -m scripts.build_submission --config config.kaggle.yaml --challenges data/arc-agi-2/evaluation_challenges.json --limit 1 --out submission.json
```

Note: `qwen2.5:1.5b` is for plumbing smoke-tests only. Target model for Kaggle is **Qwen3-8B 4-bit** via `kaggle_local`.

## Project layout

See the technical build spec. Entry point for one puzzle:

```python
from src.pipeline import solve_from_config
guesses = solve_from_config("data/arc-agi-2/training/6150a2bd.json")
```
