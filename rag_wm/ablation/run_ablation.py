"""
run_ablation.py
───────────────
Comprehensive, publication-ready ablation study suite for RAG-WM.

Evaluates 5 critical design dimensions:
  1. Gating & Residual Controls:
     - Baseline (Pure JEPA-WM, K=0, λ=0)
     - Static Residual Control (Global empirical momentum delta, no retrieval)
     - Fixed Blend Weight (Constant λ)
     - Geometric Horizon Decay (λ_t = λ_0 * γ^t)
     - Adaptive Confidence Gating (λ_t gated by retrieval cosine similarity)
     - Full RAG-WM (Adaptive Gating + Horizon Decay)
  2. Retrieval Neighborhood Size (K ∈ {1, 2, 3, 5, 8, 10} vs K=0)
  3. Memory Blending Weight (λ ∈ {0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0})
  4. Memory Bank Capacity Scaling (Ratios ∈ {10%, 25%, 50%, 75%, 100%})
  5. Softmax Temperature (τ ∈ {0.02, 0.05, 0.10, 0.20, 0.50, 1.00})

Outputs:
  All artifacts (high-res 300 DPI figures in PNG, structured JSON,
  and markdown summaries) are saved in the environment's subfolder:
      rag_wm/ablation/<env>/

Usage:
  python rag_wm/ablation/run_ablation.py --env pointmaze
  python rag_wm/ablation/run_ablation.py --env pointmaze --quick
  python rag_wm/ablation/run_ablation.py --env metaworld --ablation gating
  python rag_wm/ablation/run_ablation.py --env pusht --ablation k
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

if sys.platform == "win32":
    if sys.stdout and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if sys.stderr and hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as scipy_stats
import torch
import torch.nn.functional as F
from einops import rearrange
from tensordict import TensorDict

# ── Workspace Setup ───────────────────────────────────────────────────────────
ABLATION_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = ABLATION_DIR.parent
RESEARCH_ROOT = WORKSPACE_ROOT.parent
JEPA_ROOT = RESEARCH_ROOT / "jepa-wms"

for _p in (str(WORKSPACE_ROOT), str(JEPA_ROOT), str(RESEARCH_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from agent.rag_agent import RAGWorldModelAgent
from benchmarks.benchmark_base import BenchmarkConfig
from benchmarks.metaworld_env import MetaWorldBenchmark
from benchmarks.pointmaze_env import PointMazeBenchmark
from benchmarks.pusht_env import PushTBenchmark
from memory.episodic_memory import EpisodicMemoryBank

# ── Environment Registry ──────────────────────────────────────────────────────
_ENV_REGISTRY = {
    "metaworld": ("MetaWorld", MetaWorldBenchmark),
    "pointmaze": ("PointMaze", PointMazeBenchmark),
    "pusht":     ("Push-T",    PushTBenchmark),
}

# ── Publication Plot Styling ──────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "figure.titlesize": 13,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "figure.autolayout": True,
})

_PALETTE = {
    "baseline":        "#7f7f7f",  # Neutral gray
    "static_residual": "#e66101",  # Dark orange
    "fixed_lambda":    "#2b83ba",  # Deep blue
    "horizon_decay":   "#7b3294",  # Purple
    "adaptive_gating": "#008837",  # Forest green
    "full_rag":        "#d7191c",  # Vivid crimson
    "accent1":         "#1f78b4",
    "accent2":         "#33a02c",
}


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    """Patch-wise cosine similarity: mean over token dimension -> scalar."""
    return torch.cosine_similarity(a, b, dim=-1).mean().item()


# ── Cached Preprocessed Episode ───────────────────────────────────────────────
@dataclass
class PreprocessedItem:
    ep_idx: int
    z_ctxt: TensorDict
    act_suffix: torch.Tensor       # (H, 1, act_dim)
    gt_future: torch.Tensor        # (1, H, N_tokens, D)
    horizon: int


class AblationStudyRunner:
    """Orchestrates comprehensive multi-axis ablation experiments for RAG-WM."""

    def __init__(
        self,
        env_key: str,
        horizon: int = 20,
        quick: bool = False,
        max_episodes: Optional[int] = None,
        output_dir: Optional[Path] = None,
    ):
        self.env_key = env_key.lower()
        if self.env_key not in _ENV_REGISTRY:
            raise ValueError(f"Unknown environment '{env_key}'. Choose from: {list(_ENV_REGISTRY.keys())}")

        self.env_name, self.benchmark_cls = _ENV_REGISTRY[self.env_key]
        self.horizon = horizon
        self.quick = quick
        self.max_episodes = max_episodes
        self.output_dir = Path(output_dir) if output_dir else (ABLATION_DIR / self.env_key)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.cfg = BenchmarkConfig(
            env_name=self.env_name,
            horizon=self.horizon,
            output_dir=self.output_dir,
        )

        self.agent: Optional[RAGWorldModelAgent] = None
        self.benchmark: Optional[Any] = None
        self.cached_episodes: List[PreprocessedItem] = []

    def setup(self) -> None:
        """Loads environment model, memory bank, and pre-encodes test episodes into GPU cache."""
        print("\n" + "=" * 80)
        print(f"  INITIALIZING RAG-WM ABLATION SUITE: {self.env_name.upper()}")
        print(f"  Target Device: {self.device} | Rollout Horizon: {self.horizon}")
        print("=" * 80)

        self.benchmark = self.benchmark_cls(self.cfg)
        model, preprocessor = self.benchmark.load_model()

        self.agent = RAGWorldModelAgent(
            base_model=model,
            preprocessor=preprocessor,
            rag_lambda=self.cfg.rag_lambda,
            device=self.device,
        )

        print(f"\n[1/3] Building Episodic Memory Bank for {self.env_name}...")
        t0 = time.time()
        self.agent.memory_bank = self.benchmark.build_memory_bank(self.agent)
        print(f"      Episodic Memory Bank ready in {time.time() - t0:.2f}s ({self.agent.memory_bank.num_transitions} transitions).")

        print(f"\n[2/3] Preprocessing and Caching Held-Out Test Trajectories...")
        test_episodes = self.benchmark.get_test_episodes()
        if self.quick:
            target_n = min(4, len(test_episodes))
            test_episodes = test_episodes[:target_n]
            print(f"      [QUICK MODE ACTIVATED] Using subset of {target_n} test trajectories.")
        elif self.max_episodes is not None:
            target_n = min(self.max_episodes, len(test_episodes))
            test_episodes = test_episodes[:target_n]
            print(f"      Limiting to {target_n} test trajectories.")

        self.cached_episodes = []
        with torch.no_grad():
            for ep_idx, ep_data in enumerate(test_episodes):
                pre = self.benchmark.preprocess_episode(ep_data, self.agent)
                if pre is None:
                    continue

                obs_dict = pre["obs_dict"]
                act_suffix = pre["act_suffix"]
                gt_latents = self.agent.model.encode(obs_dict)
                gt_patches = rearrange(gt_latents["visual"], "b t v h w d -> b t (v h w) d")
                gt_future = gt_patches[:, self.cfg.ctxt_window:]

                z_ctxt = TensorDict(
                    {
                        "visual": gt_latents["visual"][:, : self.cfg.ctxt_window],
                        "proprio": gt_latents["proprio"][:, : self.cfg.ctxt_window],
                    },
                    batch_size=[1],
                )

                self.cached_episodes.append(
                    PreprocessedItem(
                        ep_idx=ep_idx,
                        z_ctxt=z_ctxt,
                        act_suffix=act_suffix,
                        gt_future=gt_future,
                        horizon=self.horizon,
                    )
                )

        print(f"      Successfully cached {len(self.cached_episodes)} preprocessed test episodes on {self.device}.")
        print("[3/3] Ready to execute ablation evaluations.\n")

    # ── Modular Rollout Evaluation ─────────────────────────────────────────────
    @torch.no_grad()
    def _evaluate_configuration(
        self,
        config_name: str,
        rollout_fn: Any,
        memory_bank: Optional[EpisodicMemoryBank] = None,
    ) -> Dict[str, Any]:
        """
        Executes a custom rollout across all cached test episodes and computes statistics.
        """
        original_bank = self.agent.memory_bank
        if memory_bank is not None:
            self.agent.memory_bank = memory_bank

        per_step_mses: List[List[float]] = [[] for _ in range(self.horizon)]
        per_step_coss: List[List[float]] = [[] for _ in range(self.horizon)]
        episode_mean_mses: List[float] = []

        for item in self.cached_episodes:
            pred_patches = rollout_fn(item)
            pred_future = pred_patches[:, self.cfg.ctxt_window:]

            ep_mses = []
            for h in range(self.horizon):
                gt_h = item.gt_future[0, h]
                pred_h = pred_future[0, h]
                mse = torch.mean((pred_h - gt_h) ** 2).item()
                cos = _cosine(pred_h, gt_h)
                per_step_mses[h].append(mse)
                per_step_coss[h].append(cos)
                ep_mses.append(mse)

            episode_mean_mses.append(float(np.mean(ep_mses)))

        # Restore original memory bank
        if memory_bank is not None:
            self.agent.memory_bank = original_bank

        step_stats = []
        for h in range(self.horizon):
            ms = per_step_mses[h]
            cs = per_step_coss[h]
            step_stats.append({
                "step": h + 1,
                "mse_mean": float(np.mean(ms)),
                "mse_std":  float(np.std(ms)),
                "mse_sem":  float(scipy_stats.sem(ms)) if len(ms) > 1 else 0.0,
                "cos_mean": float(np.mean(cs)),
            })

        mean_horizon_mse = float(np.mean(episode_mean_mses))
        final_step_mse = step_stats[-1]["mse_mean"]

        return {
            "config_name": config_name,
            "mean_horizon_mse": mean_horizon_mse,
            "final_step_mse": final_step_mse,
            "step_stats": step_stats,
            "episode_mses": episode_mean_mses,
        }

    # ── Ablation 1: Gating Mechanisms & Control Baselines ─────────────────────
    def run_ablation_gating(self) -> Dict[str, Any]:
        """
        Compares:
          - Baseline (JEPA-WM, K=0, λ=0)
          - Static Residual Control (constant empirical momentum delta)
          - Fixed Blend Weight (λ=0.30, no gating)
          - Geometric Horizon Decay (λ_0=0.30, γ=0.95)
          - Adaptive Confidence Gating (τ=0.96)
          - Full RAG-WM (Adaptive Gating + Horizon Decay)
        """
        print("─" * 80)
        print("  ABLATION 1: GATING MECHANISMS & RESIDUAL CONTROLS")
        print("─" * 80)

        configs = [
            (
                "baseline",
                "Baseline (JEPA-WM)",
                lambda item: self.agent.rollout_baseline(item.z_ctxt, item.act_suffix),
            ),
            (
                "static_residual",
                "Static Residual Control",
                lambda item: self.agent.rollout_static_residual(
                    item.z_ctxt, item.act_suffix, rag_lambda=self.cfg.rag_lambda
                ),
            ),
            (
                "fixed_lambda",
                r"Fixed Blend ($\lambda=0.30$)",
                lambda item: self.agent.rollout_rag(
                    item.z_ctxt, item.act_suffix,
                    top_k=self.cfg.top_k,
                    rag_lambda=self.cfg.rag_lambda,
                    temperature=self.cfg.temperature,
                    adaptive_gating=False,
                )["pred_patches"],
            ),
            (
                "horizon_decay",
                r"Horizon Decay ($\gamma=0.95$)",
                lambda item: self.agent.rollout_rag(
                    item.z_ctxt, item.act_suffix,
                    top_k=self.cfg.top_k,
                    rag_lambda=self.cfg.rag_lambda,
                    temperature=self.cfg.temperature,
                    adaptive_gating=False,
                    horizon_decay=self.cfg.horizon_decay,
                )["pred_patches"],
            ),
            (
                "adaptive_gating",
                "Adaptive Confidence Gating",
                lambda item: self.agent.rollout_rag(
                    item.z_ctxt, item.act_suffix,
                    top_k=self.cfg.top_k,
                    rag_lambda=self.cfg.rag_lambda,
                    temperature=self.cfg.temperature,
                    adaptive_gating=True,
                    similarity_threshold=self.cfg.similarity_threshold,
                )["pred_patches"],
            ),
            (
                "full_rag",
                "Full RAG-WM (Adaptive + Decay)",
                lambda item: self.agent.rollout_rag(
                    item.z_ctxt, item.act_suffix,
                    top_k=self.cfg.top_k,
                    rag_lambda=self.cfg.rag_lambda,
                    temperature=self.cfg.temperature,
                    adaptive_gating=True,
                    similarity_threshold=self.cfg.similarity_threshold,
                    horizon_decay=self.cfg.horizon_decay,
                )["pred_patches"],
            ),
        ]

        results = {}
        for key, label, fn in configs:
            t0 = time.time()
            res = self._evaluate_configuration(label, fn)
            res["key"] = key
            results[key] = res
            print(f"  • {label:<32}: Mean MSE = {res['mean_horizon_mse']:.4f} | Final MSE = {res['final_step_mse']:.4f} ({time.time() - t0:.2f}s)")

        # Statistical comparisons against baseline
        base_mses = np.array(results["baseline"]["episode_mses"])
        for key, res in results.items():
            if key == "baseline":
                res["gain_pct"] = 0.0
                res["p_value"] = 1.0
                res["cohens_d"] = 0.0
                continue
            cur_mses = np.array(res["episode_mses"])
            diff = base_mses - cur_mses
            gain = (float(np.mean(base_mses)) - float(np.mean(cur_mses))) / float(np.mean(base_mses)) * 100.0
            _, p_val = scipy_stats.ttest_rel(base_mses, cur_mses)
            cd = float(np.mean(diff) / (np.std(diff, ddof=1) + 1e-8))
            res["gain_pct"] = gain
            res["p_value"] = float(p_val)
            res["cohens_d"] = cd

        return results

    # ── Ablation 2: Number of Neighbors K ─────────────────────────────────────
    def run_ablation_k(self, k_values: List[int] = [1, 2, 3, 5, 8, 10]) -> Dict[str, Any]:
        """Evaluates sensitivity to retrieval neighborhood size K."""
        print("\n" + "─" * 80)
        print(f"  ABLATION 2: SENSITIVITY TO RETRIEVAL NEIGHBORS (K ∈ {k_values})")
        print("─" * 80)

        results = {}
        # Include K=0 as baseline anchor
        res_0 = self._evaluate_configuration(
            "K=0 (Baseline)",
            lambda item: self.agent.rollout_baseline(item.z_ctxt, item.act_suffix),
        )
        res_0["k"] = 0
        results[0] = res_0
        print(f"  • K=0 (Baseline)   : Mean MSE = {res_0['mean_horizon_mse']:.4f} | Final MSE = {res_0['final_step_mse']:.4f}")

        for k in k_values:
            t0 = time.time()
            res = self._evaluate_configuration(
                f"K={k}",
                lambda item, k_val=k: self.agent.rollout_rag(
                    item.z_ctxt, item.act_suffix,
                    top_k=k_val,
                    rag_lambda=self.cfg.rag_lambda,
                    temperature=self.cfg.temperature,
                    adaptive_gating=False,
                )["pred_patches"],
            )
            res["k"] = k
            results[k] = res
            print(f"  • K={k:<2}            : Mean MSE = {res['mean_horizon_mse']:.4f} | Final MSE = {res['final_step_mse']:.4f} ({time.time() - t0:.2f}s)")

        return results

    # ── Ablation 3: Blending Weight λ ─────────────────────────────────────────
    def run_ablation_lambda(
        self,
        lambdas: List[float] = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0],
    ) -> Dict[str, Any]:
        """Evaluates trade-off between neural prior (λ=0) and episodic grounding (λ=1)."""
        print("\n" + "─" * 80)
        print(f"  ABLATION 3: SENSITIVITY TO BLEND WEIGHT λ ∈ {lambdas}")
        print("─" * 80)

        results = {}
        for lam in lambdas:
            t0 = time.time()
            if lam == 0.0:
                res = self._evaluate_configuration(
                    "λ=0.0 (Pure Model)",
                    lambda item: self.agent.rollout_baseline(item.z_ctxt, item.act_suffix),
                )
            else:
                res = self._evaluate_configuration(
                    f"λ={lam:.2f}",
                    lambda item, l_val=lam: self.agent.rollout_rag(
                        item.z_ctxt, item.act_suffix,
                        top_k=self.cfg.top_k,
                        rag_lambda=l_val,
                        temperature=self.cfg.temperature,
                        adaptive_gating=False,
                    )["pred_patches"],
                )
            res["lambda"] = lam
            results[lam] = res
            print(f"  • λ={lam:<4.2f}          : Mean MSE = {res['mean_horizon_mse']:.4f} | Final MSE = {res['final_step_mse']:.4f} ({time.time() - t0:.2f}s)")

        return results

    # ── Ablation 4: Memory Bank Scaling ───────────────────────────────────────
    def run_ablation_memory(
        self,
        ratios: List[float] = [0.10, 0.25, 0.50, 0.75, 1.00],
    ) -> Dict[str, Any]:
        """Evaluates how rollout fidelity scales with episodic experience volume."""
        print("\n" + "─" * 80)
        print(f"  ABLATION 4: MEMORY CAPACITY SCALING (Buffer Ratios ∈ {ratios})")
        print("─" * 80)

        total_trans = self.agent.memory_bank.num_transitions
        results = {}

        for r in ratios:
            sub_size = max(10, int(r * total_trans))
            sub_bank = self.agent.memory_bank.get_sliced_subview(sub_size)

            t0 = time.time()
            res = self._evaluate_configuration(
                f"{int(r*100)}% ({sub_size} trans)",
                lambda item: self.agent.rollout_rag(
                    item.z_ctxt, item.act_suffix,
                    top_k=self.cfg.top_k,
                    rag_lambda=self.cfg.rag_lambda,
                    temperature=self.cfg.temperature,
                    adaptive_gating=False,
                )["pred_patches"],
                memory_bank=sub_bank,
            )
            res["ratio"] = r
            res["num_transitions"] = sub_size
            results[r] = res
            print(f"  • Ratio {r*100:>3.0f}% ({sub_size:>5d} trans): Mean MSE = {res['mean_horizon_mse']:.4f} | Final MSE = {res['final_step_mse']:.4f} ({time.time() - t0:.2f}s)")

        return results

    # ── Ablation 5: Softmax Temperature τ ─────────────────────────────────────
    def run_ablation_temperature(
        self,
        temperatures: List[float] = [0.02, 0.05, 0.10, 0.20, 0.50, 1.00],
    ) -> Dict[str, Any]:
        """Evaluates sensitivity to retrieval softmax temperature τ."""
        print("\n" + "─" * 80)
        print(f"  ABLATION 5: SOFTMAX TEMPERATURE SENSITIVITY (τ ∈ {temperatures})")
        print("─" * 80)

        results = {}
        for tau in temperatures:
            t0 = time.time()
            res = self._evaluate_configuration(
                f"τ={tau:.2f}",
                lambda item, t_val=tau: self.agent.rollout_rag(
                    item.z_ctxt, item.act_suffix,
                    top_k=self.cfg.top_k,
                    rag_lambda=self.cfg.rag_lambda,
                    temperature=t_val,
                    adaptive_gating=False,
                )["pred_patches"],
            )
            res["temperature"] = tau
            results[tau] = res
            print(f"  • τ={tau:<4.2f}          : Mean MSE = {res['mean_horizon_mse']:.4f} | Final MSE = {res['final_step_mse']:.4f} ({time.time() - t0:.2f}s)")

        return results

    # ── Publication Figure Generation (300 DPI, PNG + PDF) ────────────────────
    def save_publication_figures(
        self,
        gating_res: Optional[Dict[str, Any]] = None,
        k_res: Optional[Dict[str, Any]] = None,
        lam_res: Optional[Dict[str, Any]] = None,
        mem_res: Optional[Dict[str, Any]] = None,
    ) -> List[Path]:
        """
        Creates publication-quality 4-panel overview figure and component gain bar chart.
        Saves both 300 DPI PNG and vector PDF for inclusion in LaTeX papers.
        """
        env_slug = self.env_name.lower().replace("-", "_").replace(" ", "_")
        saved_paths: List[Path] = []

        # ── Figure 1: 4-Panel Overview ────────────────────────────────────────
        fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.5), dpi=300)
        fig.suptitle(
            f"{self.env_name}: RAG-WM Systematic Ablation Study (H={self.horizon}, N={len(self.cached_episodes)} held-out rollouts)",
            fontweight="bold",
            fontsize=13,
            y=0.995,
        )

        # (a) Gating mechanisms over horizon
        ax = axes[0, 0]
        if gating_res:
            steps = list(range(1, self.horizon + 1))
            plot_keys = [
                ("baseline",        "Baseline (JEPA-WM)",          _PALETTE["baseline"],        "-o"),
                ("static_residual", "Static Residual Control",     _PALETTE["static_residual"], "--x"),
                ("fixed_lambda",    r"Fixed Blend ($\lambda=0.30$)", _PALETTE["fixed_lambda"],   "-s"),
                ("horizon_decay",   r"Decay ($\gamma=0.95$)",        _PALETTE["horizon_decay"],  "-d"),
                ("adaptive_gating", "Adaptive Gating",             _PALETTE["adaptive_gating"], "-^"),
                ("full_rag",        "Full RAG-WM",                 _PALETTE["full_rag"],        "-*"),
            ]
            for key, label, col, fmt in plot_keys:
                if key in gating_res:
                    res = gating_res[key]
                    ms = [s["mse_mean"] for s in res["step_stats"]]
                    se = [s["mse_sem"]  for s in res["step_stats"]]
                    ax.plot(steps, ms, fmt, color=col, label=label, lw=1.8, markersize=5)
                    ax.fill_between(steps, np.array(ms) - np.array(se), np.array(ms) + np.array(se), color=col, alpha=0.12)
            ax.set_title("(a) Multi-Step Latent MSE over Rollout Horizon", fontweight="bold")
            ax.set_xlabel("Horizon Step (H)")
            ax.set_ylabel("Latent Patch MSE (↓)")
            ax.legend(frameon=True, facecolor="white", edgecolor="#e0e0e0", loc="upper left")
            ax.set_xticks(steps[::2] if len(steps) > 10 else steps)
        else:
            ax.text(0.5, 0.5, "Gating ablation not evaluated", ha="center", va="center")

        # (b) Sensitivity to K
        ax = axes[0, 1]
        if k_res:
            k_vals = sorted(k_res.keys())
            mean_mses = [k_res[k]["mean_horizon_mse"] for k in k_vals]
            final_mses = [k_res[k]["final_step_mse"] for k in k_vals]
            labels = [f"K={k}" if k > 0 else "Base" for k in k_vals]

            x_pos = np.arange(len(k_vals))
            width = 0.36
            ax.bar(x_pos - width/2, mean_mses, width, label="Mean Horizon MSE", color=_PALETTE["accent1"], alpha=0.85, edgecolor="white")
            ax.bar(x_pos + width/2, final_mses, width, label=f"Final Step MSE (H={self.horizon})", color=_PALETTE["static_residual"], alpha=0.85, edgecolor="white")

            ax.set_title("(b) Effect of Retrieval Neighborhood Size K", fontweight="bold")
            ax.set_xlabel("Neighborhood Size K")
            ax.set_ylabel("MSE (↓)")
            ax.set_xticks(x_pos)
            ax.set_xticklabels(labels)
            ax.legend(frameon=True, facecolor="white", edgecolor="#e0e0e0")
        else:
            ax.text(0.5, 0.5, "K sensitivity not evaluated", ha="center", va="center")

        # (c) Sensitivity to Blend Weight λ
        ax = axes[1, 0]
        if lam_res:
            l_vals = sorted(lam_res.keys())
            mean_mses = [lam_res[l]["mean_horizon_mse"] for l in l_vals]
            final_mses = [lam_res[l]["final_step_mse"] for l in l_vals]

            ax.plot(l_vals, mean_mses, "-o", color=_PALETTE["fixed_lambda"], lw=2.2, label="Mean Horizon MSE")
            ax.plot(l_vals, final_mses, "-s", color=_PALETTE["full_rag"], lw=2.0, label=f"Final Step MSE (H={self.horizon})")

            best_idx = int(np.argmin(mean_mses))
            best_l = l_vals[best_idx]
            best_val = mean_mses[best_idx]
            ax.plot(best_l, best_val, "o", markersize=11, markerfacecolor="gold", markeredgecolor="black", markeredgewidth=1.5, zorder=5)
            ax.annotate(
                f"Optimal $\\lambda={best_l:.2f}$\n(MSE: {best_val:.3f})",
                xy=(best_l, best_val),
                xytext=(best_l + 0.08, best_val + 0.15 * (max(mean_mses) - min(mean_mses))),
                arrowprops=dict(facecolor="black", shrink=0.08, width=1, headwidth=6),
                fontsize=9,
                fontweight="bold",
            )

            ax.set_title(r"(c) Blending Parameter $\lambda$ Convex Trade-Off Curve", fontweight="bold")
            ax.set_xlabel(r"Interpolation Weight $\lambda$ ($0 = \text{Pure Model}, 1 = \text{Pure Memory}$)")
            ax.set_ylabel("MSE (↓)")
            ax.legend(frameon=True, facecolor="white", edgecolor="#e0e0e0")
            ax.set_xticks(l_vals)
        else:
            ax.text(0.5, 0.5, "Lambda ablation not evaluated", ha="center", va="center")

        # (d) Memory Capacity Scaling
        ax = axes[1, 1]
        if mem_res:
            ratios = sorted(mem_res.keys())
            trans = [mem_res[r]["num_transitions"] for r in ratios]
            mean_mses = [mem_res[r]["mean_horizon_mse"] for r in ratios]

            ax.plot(trans, mean_mses, "-D", color=_PALETTE["adaptive_gating"], lw=2.2, markersize=7, label="RAG-WM Scaling")
            for t, m, r in zip(trans, mean_mses, ratios):
                ax.annotate(f"{int(r*100)}%", (t, m), textcoords="offset points", xytext=(0, 9), ha="center", fontsize=9, fontweight="bold")

            ax.set_title("(d) Memory Bank Capacity Scaling Law", fontweight="bold")
            ax.set_xlabel("Number of Indexed Transitions in Episodic Bank")
            ax.set_ylabel("Mean Horizon MSE (↓)")
            ax.legend(frameon=True, facecolor="white", edgecolor="#e0e0e0")
        else:
            ax.text(0.5, 0.5, "Memory scaling not evaluated", ha="center", va="center")

        plt.tight_layout()

        # Save Figure 1 (PNG only, 300 DPI)
        png_path = self.output_dir / f"{env_slug}_ablation_overview.png"
        fig.savefig(png_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        saved_paths.append(png_path)
        print(f"\n[INFO] Saved overview figure to:\n       • {png_path}")

        # ── Figure 2: Component Gain Bar Chart ────────────────────────────────
        if gating_res:
            fig2, ax2 = plt.subplots(figsize=(9, 4.5), dpi=300)
            comp_keys = ["static_residual", "fixed_lambda", "horizon_decay", "adaptive_gating", "full_rag"]
            labels = [gating_res[k]["config_name"] for k in comp_keys if k in gating_res]
            gains = [gating_res[k]["gain_pct"] for k in comp_keys if k in gating_res]
            p_vals = [gating_res[k].get("p_value", 1.0) for k in comp_keys if k in gating_res]
            colors = [
                _PALETTE["static_residual"],
                _PALETTE["fixed_lambda"],
                _PALETTE["horizon_decay"],
                _PALETTE["adaptive_gating"],
                _PALETTE["full_rag"],
            ][:len(labels)]

            y_pos = np.arange(len(labels))
            bars = ax2.barh(y_pos, gains, color=colors, alpha=0.85, edgecolor="white", height=0.55)
            ax2.axvline(0, color="black", lw=0.9, ls="--")

            for bar, gain, p in zip(bars, gains, p_vals):
                sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
                sign_str = "+" if gain >= 0 else ""
                text = f" {sign_str}{gain:.2f}% ({sig})"
                ax2.text(max(0, bar.get_width()) + 0.3, bar.get_y() + bar.get_height()/2, text, va="center", fontsize=9, fontweight="bold")

            ax2.set_yticks(y_pos)
            ax2.set_yticklabels(labels)
            ax2.set_xlabel("Relative Latent MSE Reduction vs JEPA-WM Baseline (%) [Higher is Better ↑]")
            ax2.set_title(f"{self.env_name}: Component Contribution to Latent Prediction Quality", fontweight="bold")
            plt.tight_layout()

            bar_png = self.output_dir / f"{env_slug}_ablation_components_bar.png"
            fig2.savefig(bar_png, dpi=300, bbox_inches="tight")
            plt.close(fig2)
            saved_paths.append(bar_png)
            print(f"[INFO] Saved component bar figure to:\n       • {bar_png}")

        return saved_paths

    # ── Save Structured Results JSON & Markdown ───────────────────────────────
    def save_results(
        self,
        all_ablation_data: Dict[str, Any],
    ) -> Tuple[Path, Path]:
        """Saves complete structured JSON and human-readable Markdown summary in results/."""
        env_slug = self.env_name.lower().replace("-", "_").replace(" ", "_")
        json_path = self.output_dir / f"{env_slug}_ablation_results.json"
        md_path = self.output_dir / f"{env_slug}_ablation_summary.md"

        # Serialize to JSON
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(all_ablation_data, f, indent=2)
        print(f"[INFO] Saved structured results JSON to:\n       • {json_path}")

        # Generate clean Markdown summary
        md_lines = [
            f"# Systematic Ablation Study: {self.env_name}",
            f"- **Rollout Horizon**: {self.horizon}",
            f"- **Evaluated Held-Out Trajectories**: {len(self.cached_episodes)}",
            f"- **Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## 1. Gating & Control Baseline Ablation",
            "| Architecture Variant | Mean Horizon MSE | Final Step MSE | Gain vs Base (%) | Cohen's d | p-Value | Sig |",
            "|:---|:---:|:---:|:---:|:---:|:---:|:---:|",
        ]

        gating = all_ablation_data.get("gating", {})
        for key, res in gating.items():
            name = res.get("config_name", key)
            m_mse = res.get("mean_horizon_mse", 0.0)
            f_mse = res.get("final_step_mse", 0.0)
            gain = res.get("gain_pct", 0.0)
            cd = res.get("cohens_d", 0.0)
            pval = res.get("p_value", 1.0)
            sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "n.s."
            md_lines.append(
                f"| **{name}** | {m_mse:.4f} | {f_mse:.4f} | {gain:+.2f}% | {cd:.3f} | {pval:.2e} | {sig} |"
            )

        if "k_sensitivity" in all_ablation_data:
            md_lines.extend([
                "",
                "## 2. Retrieval Neighborhood Size (K) Sensitivity",
                "| Neighborhood Size K | Mean Horizon MSE | Final Step MSE |",
                "|:---:|:---:|:---:|",
            ])
            for k, res in all_ablation_data["k_sensitivity"].items():
                md_lines.append(f"| K={k} | {res['mean_horizon_mse']:.4f} | {res['final_step_mse']:.4f} |")

        if "lambda_sensitivity" in all_ablation_data:
            md_lines.extend([
                "",
                "## 3. Blend Weight (λ) Sensitivity",
                "| Interpolation Weight λ | Mean Horizon MSE | Final Step MSE |",
                "|:---:|:---:|:---:|",
            ])
            for lam, res in all_ablation_data["lambda_sensitivity"].items():
                md_lines.append(f"| λ={float(lam):.2f} | {res['mean_horizon_mse']:.4f} | {res['final_step_mse']:.4f} |")

        if "memory_scaling" in all_ablation_data:
            md_lines.extend([
                "",
                "## 4. Memory Bank Capacity Scaling",
                "| Experience Ratio | Transitions Indexed | Mean Horizon MSE | Final Step MSE |",
                "|:---:|:---:|:---:|:---:|",
            ])
            for r, res in all_ablation_data["memory_scaling"].items():
                md_lines.append(
                    f"| {float(r)*100:.0f}% | {res['num_transitions']} | {res['mean_horizon_mse']:.4f} | {res['final_step_mse']:.4f} |"
                )

        if "temperature_sensitivity" in all_ablation_data:
            md_lines.extend([
                "",
                "## 5. Softmax Temperature (τ) Sensitivity",
                "| Softmax Temperature τ | Mean Horizon MSE | Final Step MSE |",
                "|:---:|:---:|:---:|",
            ])
            for tau, res in all_ablation_data["temperature_sensitivity"].items():
                md_lines.append(f"| τ={float(tau):.2f} | {res['mean_horizon_mse']:.4f} | {res['final_step_mse']:.4f} |")

        with open(md_path, "w", encoding="utf-8") as f:
            f.write("\n".join(md_lines) + "\n")
        print(f"[INFO] Saved human-readable summary Markdown to:\n       • {md_path}")

        return json_path, md_path

    # ── Console Table Display ─────────────────────────────────────────────────
    def print_summary_table(self, gating_res: Dict[str, Any]) -> None:
        """Pretty-prints the core gating ablation table to stdout."""
        W = 100
        print("\n" + "=" * W)
        print(f"  ABLATION RESULTS SUMMARY: {self.env_name.upper()} (Horizon H={self.horizon})")
        print("=" * W)
        print(
            f"{'Variant':<34} | {'Mean MSE':^12} | {'Final MSE':^12} | {'Gain vs Base':^14} | {'Cohen d':^9} | {'Sig':^5}"
        )
        print("-" * W)
        for key, res in gating_res.items():
            name = res.get("config_name", key)
            m_mse = res.get("mean_horizon_mse", 0.0)
            f_mse = res.get("final_step_mse", 0.0)
            gain = res.get("gain_pct", 0.0)
            cd = res.get("cohens_d", 0.0)
            pval = res.get("p_value", 1.0)
            sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "n.s."
            sign_str = "+" if gain >= 0 else ""
            print(
                f"{name:<34} | {m_mse:10.4f}   | {f_mse:10.4f}   | {sign_str}{gain:8.2f}%    | {cd:7.3f}   | {sig:^5}"
            )
        print("=" * W + "\n")


# ── CLI Entrypoint ────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Comprehensive Ablation Study for RAG-WM Paper",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--env",
        type=str,
        default="pointmaze",
        choices=list(_ENV_REGISTRY.keys()),
        help="Environment to evaluate",
    )
    parser.add_argument(
        "--ablation",
        type=str,
        default="all",
        choices=["all", "gating", "k", "lambda", "memory", "temp"],
        help="Specific ablation study axis to run, or 'all'",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=20,
        help="Rollout horizon length H",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quickly on a small subset of test episodes for rapid iteration",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=None,
        help="Explicitly cap number of test trajectories evaluated",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save figures and results (defaults to rag_wm/ablation/<env>/)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir) if args.output_dir else (ABLATION_DIR / args.env)

    runner = AblationStudyRunner(
        env_key=args.env,
        horizon=args.horizon,
        quick=args.quick,
        max_episodes=args.num_episodes,
        output_dir=out_dir,
    )
    runner.setup()

    all_data: Dict[str, Any] = {
        "environment": runner.env_name,
        "horizon": runner.horizon,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    gating_res = None
    k_res = None
    lam_res = None
    mem_res = None

    run_all = args.ablation == "all"

    if run_all or args.ablation == "gating":
        gating_res = runner.run_ablation_gating()
        all_data["gating"] = gating_res
        runner.print_summary_table(gating_res)

    if run_all or args.ablation == "k":
        k_res = runner.run_ablation_k()
        all_data["k_sensitivity"] = {str(k): v for k, v in k_res.items()}

    if run_all or args.ablation == "lambda":
        lam_res = runner.run_ablation_lambda()
        all_data["lambda_sensitivity"] = {str(l): v for l, v in lam_res.items()}

    if run_all or args.ablation == "memory":
        mem_res = runner.run_ablation_memory()
        all_data["memory_scaling"] = {str(r): v for r, v in mem_res.items()}

    if run_all or args.ablation == "temp":
        temp_res = runner.run_ablation_temperature()
        all_data["temperature_sensitivity"] = {str(t): v for t, v in temp_res.items()}

    # Generate and save publication figures (PNG + PDF)
    runner.save_publication_figures(
        gating_res=gating_res,
        k_res=k_res,
        lam_res=lam_res,
        mem_res=mem_res,
    )

    # Save JSON and Markdown summary
    runner.save_results(all_data)
    print("\n[✓] Ablation Study Pipeline Completed Successfully!")


if __name__ == "__main__":
    main()
