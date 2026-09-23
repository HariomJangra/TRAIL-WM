# TRAIL-WM — JEPA World Model Benchmark Suite

Clean, reviewer-ready benchmark code for the TRAIL-WM paper (*Transition Retrieval and Adaptive Integration for JEPA-Based World Models*).

---

## File Structure

```
benchmarks/
  benchmark_base.py   ← Abstract base: shared pipeline + figures + stats
  metaworld_env.py    ← MetaWorld adapter
  pointmaze_env.py    ← PointMaze adapter
  pusht_env.py        ← Push-T adapter
  run_benchmark.py    ← Unified CLI entry point
  results/            ← All outputs (JSON + PNG) written here
```

---

## Running

```bash
# from the repo root
python rag_wm/benchmarks/run_benchmark.py --env metaworld
python rag_wm/benchmarks/run_benchmark.py --env pointmaze
python rag_wm/benchmarks/run_benchmark.py --env pusht

# optional overrides
python rag_wm/benchmarks/run_benchmark.py --env pusht --horizon 20 --top-k 3 --lam 0.30
```

Or run each environment directly:

```bash
python rag_wm/benchmarks/metaworld_env.py
python rag_wm/benchmarks/pointmaze_env.py
python rag_wm/benchmarks/pusht_env.py
```

---

## Outputs (identical for every environment)

| File | Description |
|------|-------------|
| `results/<env>_benchmark_data.json` | Per-step MSE, cosine sim, gain %, Cohen's d, p-values |
| `results/<env>_benchmark_figure.png` | 2×2 figure: MSE curve, cosine curve, gain bar chart, effect size |

---

## Architecture

All three environments share **identical** evaluation logic via the `EnvironmentBenchmark`
base class (Template Method pattern). Each environment subclass implements only:

| Method | What it does |
|--------|-------------|
| `load_model()` | Load JEPA-WM model + preprocessor from checkpoint |
| `build_memory_bank()` | Index training transitions into `EpisodicMemoryBank` |
| `get_test_episodes()` | Return held-out test episodes |
| `preprocess_episode()` | Convert raw episode → model-ready tensors |

The following are **shared** and never duplicated:

- `evaluate_episode()` — baseline + fixed-TRAIL + adaptive-TRAIL rollouts
- `aggregate_stats()` — mean/std/SEM, paired t-test, Cohen's d
- `print_table()` — console output
- `save_json()` — structured JSON
- `save_figures()` — 2×2 matplotlib figure

---

## Compared Methods

| Method | Description |
|--------|-------------|
| **JEPA-WM Baseline** | Standard parametric rollout, no retrieval |
| **TRAIL-WM (fixed λ)** | Fixed blend weight λ=0.30, K=3 neighbours |
| **TRAIL-WM (adaptive gate)** | Similarity-gated λ with horizon decay |
