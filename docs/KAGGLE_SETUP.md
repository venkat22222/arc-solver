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
python -m scripts.smoke_kaggle_local --model Qwen/Qwen3-8B --max-tokens 32
```

Uses `config.yaml` `max_memory` CPU offload on 4GB GPUs. On Kaggle P100 use `config.kaggle.yaml` (no offload).
