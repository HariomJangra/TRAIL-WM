"""
run_benchmark.py
────────────────
Unified CLI entry point for all RAG-WM environment benchmarks.

Usage (run from e:\\Research):
    python rag_wm/benchmarks/run_benchmark.py --env metaworld
    python rag_wm/benchmarks/run_benchmark.py --env pointmaze
    python rag_wm/benchmarks/run_benchmark.py --env pusht

Optional overrides:
    --horizon    20
    --top-k      3
    --lam        0.30
    --temperature 0.10

All outputs (JSON + figures) are saved to:
    rag_wm/benchmarks/results/<env>_benchmark_data.json
    rag_wm/benchmarks/results/<env>_benchmark_figure.png
"""

import argparse
import sys
from pathlib import Path

if sys.platform == "win32":
    if sys.stdout and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if sys.stderr and hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from benchmarks.benchmark_base import BenchmarkConfig
from benchmarks.metaworld_env import MetaWorldBenchmark
from benchmarks.pointmaze_env import PointMazeBenchmark
from benchmarks.pusht_env import PushTBenchmark

_REGISTRY = {
    "metaworld": ("MetaWorld", MetaWorldBenchmark),
    "pointmaze": ("PointMaze", PointMazeBenchmark),
    "pusht":     ("Push-T",    PushTBenchmark),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RAG-WM Benchmark Runner")
    p.add_argument(
        "--env", required=True, choices=list(_REGISTRY),
        help="Environment to benchmark",
    )
    p.add_argument("--horizon",     type=int,   default=20,   help="Rollout horizon H")
    p.add_argument("--top-k",       type=int,   default=3,    help="Retrieval neighbours K")
    p.add_argument("--lam",         type=float, default=0.30, help="Fixed RAG blend weight λ")
    p.add_argument("--temperature", type=float, default=0.10, help="Retrieval softmax temperature τ")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    env_name, BenchmarkClass = _REGISTRY[args.env]

    cfg = BenchmarkConfig(
        env_name=env_name,
        horizon=args.horizon,
        top_k=args.top_k,
        rag_lambda=args.lam,
        temperature=args.temperature,
        output_dir=Path(__file__).resolve().parent / "results",
    )

    BenchmarkClass(cfg).run()


if __name__ == "__main__":
    main()
