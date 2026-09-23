# TRAIL-WM Package Overview

This directory contains the core implementation of **TRAIL-WM** (**T**ransition **R**etrieval and **A**daptive **I**ntegration for **L**atent **W**orld **M**odels).

## Package Layout

- **[`agent/`](agent/)**: Implements `RAGWorldModelAgent` (`agent/rag_agent.py`), orchestrating neural parametric unrolls and non-parametric episodic fusion with adaptive gating and horizon decay schedules.
- **[`memory/`](memory/)**: Implements `EpisodicMemoryBank` (`memory/episodic_memory.py`), providing GPU-accelerated cosine similarity search, capacity slicing, and temperature-scaled softmax neighbor aggregation.
- **[`benchmarks/`](benchmarks/)**: Environment evaluation suite (`metaworld_env.py`, `pointmaze_env.py`, `pusht_env.py`, `run_benchmark.py`).
- **[`ablation/`](ablation/)**: Multi-axis systematic ablation study suite (`run_ablation.py`) with per-environment results subfolders:
  - [`pointmaze/`](ablation/pointmaze/): Ablation figures, JSON data, and markdown summaries.
  - [`metaworld/`](ablation/metaworld/): MetaWorld ablation results.
  - [`pusht/`](ablation/pusht/): Push-T continuous manipulation ablation results.
- **[`paper/`](paper/)**: Complete TMLR submission paper (`main.tex`), standalone publication figures (300 DPI), style files, and bibliography.

For full usage instructions, benchmarks, and experimental details, refer to the root [README.md](../README.md).
