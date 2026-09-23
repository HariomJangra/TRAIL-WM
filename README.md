# RAG-WM: Retrieval-Augmented Joint-Embedding World Models

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**RAG-WM** enhances Joint-Embedding Predictive Architecture World Models (**JEPA-WM**) by integrating non-parametric episodic memory retrieval with parametric Transformer dynamics. By grounding long-horizon latent unrolls in observed transitions, RAG-WM significantly curtails compounding rollout drift and improves prediction fidelity across complex visual navigation and robotic manipulation tasks.

---

## Key Features

- **Dense Episodic Memory Bank**: Indexes observed transitions $(s_t, a_t \rightarrow s_{t+1}, \Delta s_t)$ in self-supervised DINOv2 latent space.
- **GPU-Accelerated Vector Retrieval**: Sub-millisecond cosine similarity search with temperature-scaled softmax multi-neighbor weighting.
- **Adaptive Confidence Gating**: Dynamically adjusts the blending parameter $\lambda_t$ according to retrieval similarity confidence to avoid negative interference from out-of-distribution transitions.
- **Horizon Decay Schedules**: Smoothly blends from memory-grounded rollouts to neural parametric rollouts over long horizons.
- **Zero Additional Training**: Plug-and-play non-parametric augmentation over frozen, pre-trained JEPA world model weights.
- **Systematic Ablation Suite**: End-to-end framework evaluating gating mechanisms, neighbor count $K$, blend ratio $\lambda$, memory bank capacity scaling, and softmax temperature $\tau$.

---

## Repository Structure

```text
rag-wm/
├── rag_wm/
│   ├── agent/                 # Agent logic combining parametric & episodic dynamics
│   │   └── rag_agent.py       # RAGWorldModelAgent implementation
│   ├── memory/                # Non-parametric episodic memory buffer
│   │   └── episodic_memory.py # GPU vector index & softmax neighbor retrieval
│   ├── benchmarks/            # Unified evaluation suite across benchmarks
│   │   ├── benchmark_base.py  # Abstract benchmark template (metrics, stats, plots)
│   │   ├── pointmaze_env.py   # PointMaze discrete 2D maze navigation adapter
│   │   ├── metaworld_env.py   # MetaWorld multi-task robotic manipulation adapter
│   │   ├── pusht_env.py       # Push-T continuous planar manipulation adapter
│   │   └── run_benchmark.py   # CLI entrypoint for standard benchmark runs
│   └── ablation/              # Systematic ablation study framework
│       ├── run_ablation.py    # Main ablation runner
│       ├── pointmaze/         # PointMaze ablation figures, data, and summaries
│       ├── metaworld/         # MetaWorld ablation folder
│       └── pusht/             # Push-T ablation folder
└── jepa-wms/                  # Underlying JEPA World Model architecture & planning tools
```

---

## Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone --recursive https://github.com/HariomJangra/rag-wm.git
   cd rag-wm

   # Or if cloned without --recursive:
   # git submodule update --init --recursive
   ```

2. **Install dependencies**:
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
   pip install tensordict einops pyarrow imageio matplotlib scipy pyyaml
   ```

---

## Running Benchmarks

Run standard evaluations comparing **JEPA-WM Baseline** vs. **Fixed-blend RAG-WM** vs. **Adaptive-gated RAG-WM**:

```bash
# PointMaze navigation
python rag_wm/benchmarks/run_benchmark.py --env pointmaze

# MetaWorld robotic manipulation
python rag_wm/benchmarks/run_benchmark.py --env metaworld

# Push-T continuous block pushing
python rag_wm/benchmarks/run_benchmark.py --env pusht
```

### Custom Overrides:
```bash
python rag_wm/benchmarks/run_benchmark.py --env pointmaze --horizon 20 --top-k 3 --lam 0.30 --temperature 0.10
```

All benchmark runs generate:
- `rag_wm/benchmarks/results/<env>_benchmark_data.json` (step-by-step MSE, cosine similarities, paired t-tests, Cohen's d)
- `rag_wm/benchmarks/results/<env>_benchmark_figure.png` (2x2 overview figure)

---

## Systematic Ablation Studies

The ablation suite in `rag_wm/ablation/run_ablation.py` systematically evaluates:
1. **Gating Mechanisms & Control Baselines**: Baseline (Pure JEPA-WM), Static Momentum Residual, Fixed $\lambda$, Horizon Decay, Adaptive Gating, and Full RAG-WM.
2. **Retrieval Neighborhood Size ($K$)**: Evaluates $K \in \{0, 1, 2, 3, 5, 8, 10\}$.
3. **Interpolation Weight ($\lambda$)**: Evaluates $\lambda \in [0.0, 1.0]$.
4. **Memory Bank Capacity Scaling**: Evaluates buffer capacity scaling across 10%, 25%, 50%, 75%, and 100% indexed transitions.
5. **Softmax Temperature ($\tau$)**: Evaluates retrieval temperature across $\tau \in [0.02, 1.00]$.

### Usage:

```bash
# Full ablation study (all 5 axes, default H=20):
python rag_wm/ablation/run_ablation.py --env pointmaze

# Fast iteration on a subset of test episodes:
python rag_wm/ablation/run_ablation.py --env pointmaze --quick

# Run a specific ablation axis:
python rag_wm/ablation/run_ablation.py --env pointmaze --ablation gating
python rag_wm/ablation/run_ablation.py --env pointmaze --ablation k
python rag_wm/ablation/run_ablation.py --env pointmaze --ablation lambda
python rag_wm/ablation/run_ablation.py --env pointmaze --ablation memory
python rag_wm/ablation/run_ablation.py --env pointmaze --ablation temp

# Run for other environments:
python rag_wm/ablation/run_ablation.py --env metaworld
python rag_wm/ablation/run_ablation.py --env pusht
```

Results are automatically saved to each experiment's dedicated subfolder (`rag_wm/ablation/<env>/`):
- `*_ablation_overview.png`: Publication-ready 4-panel multi-axis figure (300 DPI)
- `*_ablation_components_bar.png`: Component improvement bar chart with significance levels (300 DPI)
- `*_ablation_results.json`: Full numerical metrics and step-by-step stats
- `*_ablation_summary.md`: Human-readable markdown table summary

---

## Environments

| Environment | Observation Type | State Space | Action Space | Task Description |
|:---|:---:|:---:|:---:|:---|
| **PointMaze** | 224 x 224 RGB | 4D (Pos, Vel) | 2D Discrete | Long-horizon navigation around obstacles |
| **Push-T** | 224 x 224 RGB | 4D (Pos, Vel) | 2D Continuous | Planar T-block manipulation to target zone |
| **MetaWorld** | 256 x 256 RGB | 39D Arm State | 4D Continuous | Multi-task articulated robotic manipulation |

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
