# GitHub + Kaggle wiring checklist

## 1. Push this repo (one-time)

```powershell
cd c:\puzzle_solver\arc-solver
gh auth login   # browser / token — required once
gh repo create arc-solver --private --source=. --remote=origin --push
```

Or without `gh`:

```powershell
# create empty repo on github.com, then:
git remote add origin https://github.com/<you>/arc-solver.git
git push -u origin master
```

## 2. Kaggle notebook

1. Create a new **Notebook** on the ARC Prize 2025 competition (GPU, internet **on** for first setup).
2. **Add data**:
   - Competition: `arc-prize-2025`
   - Dataset: upload/push this repo as a Kaggle Dataset named e.g. `arc-solver` (must contain `src/`, `config.kaggle.yaml`, `scripts/`).
   - Optional offline model: mirror `Qwen/Qwen3-8B` as a Kaggle dataset and set `model_name` / `local_files_only` accordingly for internet-off scoring.
3. Open `notebooks/kaggle_submission.ipynb` (or copy its cells).
4. Run smoke cell → confirm `PONG`.
5. Run `scripts.build_submission` → writes `/kaggle/working/submission.json`.
6. **Save Version → Submit** (internet **off** for the scoring run if required by competition rules).

## 3. Local smoke (already scripted)

```powershell
# Full generate smoke on 4GB GPU (same kaggle_local client path as Kaggle)
python -m scripts.smoke_kaggle_local --model Qwen/Qwen3-4B --max-tokens 32

# Qwen3-8B: weights download + 4-bit load validated; generate needs >=~8–16GB VRAM
# (4-bit CPU offload is broken on Windows bitsandbytes). Use Kaggle P100:
python -m scripts.smoke_kaggle_local --model Qwen/Qwen3-8B --max-tokens 32
```

Pinned local stack that works: `torch 2.6+cu124`, `transformers 4.51.3`, `bitsandbytes 0.45.5`, `accelerate 1.2.1`.
