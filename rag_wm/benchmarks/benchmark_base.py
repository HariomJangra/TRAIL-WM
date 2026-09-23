"""
benchmark_base.py
─────────────────
Abstract base class for all RAG-WM environment benchmarks.

Every environment (MetaWorld, PointMaze, Push-T) shares an identical pipeline:

    load_model()
      → build_memory_bank()
      → get_test_episodes()
      → for each episode: evaluate_episode()
      → aggregate_stats()
      → print_table()
      → save_json()
      → save_figures()            ← all environments produce identical figures

Subclasses implement only the four environment-specific slots:
    load_model, build_memory_bank, get_test_episodes, preprocess_episode
"""

from __future__ import annotations

import json
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as scipy_stats
import torch
import torch.nn.functional as F
from einops import rearrange
from tensordict import TensorDict

# ── path setup ────────────────────────────────────────────────────────────────
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
JEPA_ROOT = WORKSPACE_ROOT.parent / "jepa-wms"
for _p in (str(WORKSPACE_ROOT), str(JEPA_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from memory.episodic_memory import EpisodicMemoryBank
from agent.rag_agent import RAGWorldModelAgent

# ── figure style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.autolayout": True,
})

_COLORS = {
    "baseline": "#d95f02",
    "rag_fix":  "#1f78b4",
    "rag_gate": "#33a02c",
}


# ── config ─────────────────────────────────────────────────────────────────────
@dataclass
class BenchmarkConfig:
    """Unified hyper-parameters for a benchmark run."""

    env_name: str                           # e.g. "MetaWorld"
    horizon: int           = 20             # rollout length H
    ctxt_window: int       = 1              # context frames fed to model
    top_k: int             = 3              # retrieval neighbours K
    rag_lambda: float      = 0.30           # fixed-RAG blend weight λ
    temperature: float     = 0.10           # retrieval softmax temperature τ
    similarity_threshold: float = 0.96     # adaptive gating threshold
    horizon_decay: float   = 0.95          # per-step λ decay for gating
    device: str            = "cuda:0"
    output_dir: Path       = field(
        default_factory=lambda: Path(__file__).resolve().parent / "results"
    )

    def __post_init__(self):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)


# ── helpers ────────────────────────────────────────────────────────────────────
def _step_result(
    step: int,
    mse_base: float, mse_fix: float, mse_gated: float,
    cos_base: float, cos_fix: float, cos_gated: float,
    sim_query: float,
) -> dict:
    """Canonical per-step result dict — same schema for every environment."""
    return dict(
        step=step,
        mse_base=mse_base, mse_fix=mse_fix, mse_gated=mse_gated,
        cos_base=cos_base, cos_fix=cos_fix, cos_gated=cos_gated,
        sim_query=sim_query,
    )


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    """Patch-wise cosine similarity: mean over token dimension → scalar."""
    return torch.cosine_similarity(a, b, dim=-1).mean().item()


# ── abstract base ─────────────────────────────────────────────────────────────
class EnvironmentBenchmark(ABC):
    """
    Template-method base for all benchmark scripts.
    Subclasses must implement the four abstract methods below.
    """

    def __init__(self, cfg: BenchmarkConfig):
        self.cfg = cfg

    # ── abstract: environment-specific ────────────────────────────────────────

    @abstractmethod
    def load_model(self) -> tuple[Any, Any]:
        """Return (model, preprocessor) for this environment."""

    @abstractmethod
    def build_memory_bank(self, agent: RAGWorldModelAgent) -> EpisodicMemoryBank:
        """Build and return an EpisodicMemoryBank from training data."""

    @abstractmethod
    def get_test_episodes(self) -> list[dict]:
        """
        Return a list of raw episode dicts.
        Each dict must have: 'frames', 'actions', 'proprios'.
        Optional: 'task' (str label for per-task breakdown).
        """

    @abstractmethod
    def preprocess_episode(
        self,
        ep_data: dict,
        agent: RAGWorldModelAgent,
    ) -> Optional[dict]:
        """
        Convert a raw episode dict into model-ready tensors.

        Must return:
            obs_dict  : {"visual": Tensor(1,T,C,H,W), "proprio": Tensor(1,T,D)}
            act_suffix: Tensor(H, 1, action_dim)
        or None if the episode is too short.
        """

    # ── shared: evaluate one episode ─────────────────────────────────────────

    @torch.no_grad()
    def evaluate_episode(
        self,
        agent: RAGWorldModelAgent,
        ep_data: dict,
    ) -> Optional[list[dict]]:
        """
        Run baseline + fixed-RAG + adaptive-RAG rollouts on one episode.
        Returns a list of per-step result dicts, or None if episode is invalid.
        """
        cfg = self.cfg
        preprocessed = self.preprocess_episode(ep_data, agent)
        if preprocessed is None:
            return None

        obs_dict   = preprocessed["obs_dict"]
        act_suffix = preprocessed["act_suffix"]   # (H, 1, action_dim)

        # Ground-truth latents
        gt_latents = agent.model.encode(obs_dict)
        gt_patches = rearrange(gt_latents["visual"], "b t v h w d -> b t (v h w) d")
        gt_future  = gt_patches[:, cfg.ctxt_window:]   # (1, H, N_tok, D)

        z_ctxt = TensorDict(
            {
                "visual": gt_latents["visual"][:, : cfg.ctxt_window],
                "proprio": gt_latents["proprio"][:, : cfg.ctxt_window],
            },
            batch_size=[1],
        )

        # 1. Parametric baseline (no retrieval)
        base_patches = agent.rollout_baseline(z_ctxt, act_suffix)
        base_future  = base_patches[:, cfg.ctxt_window:]

        # 2. Fixed-lambda RAG-WM
        rag_fix_out    = agent.rollout_rag(
            z_ctxt, act_suffix,
            top_k=cfg.top_k, rag_lambda=cfg.rag_lambda,
            temperature=cfg.temperature, adaptive_gating=False,
        )
        rag_fix_future = rag_fix_out["pred_patches"][:, cfg.ctxt_window:]

        # 3. Adaptive-gated RAG-WM
        rag_gate_out    = agent.rollout_rag(
            z_ctxt, act_suffix,
            top_k=cfg.top_k, rag_lambda=cfg.rag_lambda,
            temperature=cfg.temperature, adaptive_gating=True,
            similarity_threshold=cfg.similarity_threshold,
            horizon_decay=cfg.horizon_decay,
        )
        rag_gate_future = rag_gate_out["pred_patches"][:, cfg.ctxt_window:]

        # Per-step metrics
        results = []
        for h in range(cfg.horizon):
            gt_h    = gt_future[0, h]
            base_h  = base_future[0, h]
            fix_h   = rag_fix_future[0, h]
            gated_h = rag_gate_future[0, h]

            sim_q = (
                rag_fix_out["similarity_scores"][h]
                if len(rag_fix_out["similarity_scores"]) > h
                else 1.0
            )

            results.append(_step_result(
                step=h + 1,
                mse_base=torch.mean((base_h  - gt_h) ** 2).item(),
                mse_fix= torch.mean((fix_h   - gt_h) ** 2).item(),
                mse_gated=torch.mean((gated_h - gt_h) ** 2).item(),
                cos_base= _cosine(base_h,  gt_h),
                cos_fix=  _cosine(fix_h,   gt_h),
                cos_gated=_cosine(gated_h, gt_h),
                sim_query=sim_q,
            ))

        return results

    # ── shared: aggregate statistics ─────────────────────────────────────────

    def aggregate_stats(self, all_results: list[list[dict]]) -> list[dict]:
        """Compute mean/std/SEM, paired t-test, and Cohen's d for each step."""
        stats = []
        for h in range(self.cfg.horizon):
            base_mses  = [ep[h]["mse_base"]  for ep in all_results]
            fix_mses   = [ep[h]["mse_fix"]   for ep in all_results]
            gated_mses = [ep[h]["mse_gated"] for ep in all_results]
            base_cos   = [ep[h]["cos_base"]  for ep in all_results]
            fix_cos    = [ep[h]["cos_fix"]   for ep in all_results]
            gated_cos  = [ep[h]["cos_gated"] for ep in all_results]
            sims       = [ep[h]["sim_query"] for ep in all_results]

            m_base, s_base    = np.mean(base_mses),  np.std(base_mses)
            m_fix, s_fix      = np.mean(fix_mses),   np.std(fix_mses)
            m_gated, s_gated  = np.mean(gated_mses), np.std(gated_mses)

            diff     = np.array(base_mses) - np.array(fix_mses)
            _, p_val = scipy_stats.ttest_rel(base_mses, fix_mses)
            cohens_d = float(np.mean(diff) / (np.std(diff, ddof=1) + 1e-8))
            gain_pct = (float(m_base) - float(m_fix)) / float(m_base) * 100.0

            sig = ("***" if p_val < 0.001 else "**" if p_val < 0.01
                   else "*" if p_val < 0.05 else "n.s.")

            stats.append({
                "step":           h + 1,
                "base_mse_mean":  float(m_base),
                "base_mse_std":   float(s_base),
                "base_mse_sem":   float(scipy_stats.sem(base_mses)),
                "fix_mse_mean":   float(m_fix),
                "fix_mse_std":    float(s_fix),
                "fix_mse_sem":    float(scipy_stats.sem(fix_mses)),
                "gated_mse_mean": float(m_gated),
                "gated_mse_std":  float(s_gated),
                "gated_mse_sem":  float(scipy_stats.sem(gated_mses)),
                "gain_pct":       float(gain_pct),
                "cohens_d":       cohens_d,
                "p_value":        float(p_val),
                "significance":   sig,
                "base_cos_mean":  float(np.mean(base_cos)),
                "fix_cos_mean":   float(np.mean(fix_cos)),
                "gated_cos_mean": float(np.mean(gated_cos)),
                "mean_sim":       float(np.mean(sims)),
            })
        return stats

    # ── shared: console table ─────────────────────────────────────────────────

    def print_table(self, stats: list[dict]) -> None:
        W = 95
        print("\n" + "=" * W)
        print(
            f"{'Step':^6} | {'Baseline MSE':^18} | "
            f"{'RAG-WM (fixed) MSE':^18} | {'Gain (%)':^10} | "
            f"{'Cohen d':^9} | {'p-Value':^12} | {'Sig':^6}"
        )
        print("-" * W)
        for s in stats:
            print(
                f"Step {s['step']:2d} | "
                f"{s['base_mse_mean']:7.4f} ± {s['base_mse_std']:6.4f} | "
                f"{s['fix_mse_mean']:7.4f} ± {s['fix_mse_std']:6.4f} | "
                f"{s['gain_pct']:+8.2f}% | "
                f"{s['cohens_d']:8.3f} | "
                f"{s['p_value']:11.2e} | "
                f"{s['significance']:^6}"
            )
        print("=" * W)

    # ── shared: save JSON ────────────────────────────────────────────────────

    def save_json(self, stats: list[dict], extra: dict | None = None) -> Path:
        """Write per-step stats (plus any env-specific extras) to JSON."""
        env_slug = self.cfg.env_name.lower().replace("-", "_").replace(" ", "_")
        path = self.cfg.output_dir / f"{env_slug}_benchmark_data.json"
        payload: dict = {
            "environment":  self.cfg.env_name,
            "horizon":      self.cfg.horizon,
            "rag_lambda":   self.cfg.rag_lambda,
            "top_k":        self.cfg.top_k,
            "steps":        stats,
        }
        if extra:
            payload.update(extra)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"[INFO] Saved JSON to: {path}")
        return path

    # ── shared: save figures ─────────────────────────────────────────────────

    def save_figures(self, stats: list[dict], N: int) -> list[Path]:
        """
        Produce a 2×2 figure for every environment:
            (a) Latent MSE over horizon
            (b) Cosine similarity over horizon
            (c) Relative MSE gain (%) over horizon
            (d) Cohen's d effect size over horizon
        """
        env      = self.cfg.env_name
        env_slug = env.lower().replace("-", "_").replace(" ", "_")
        steps    = [s["step"] for s in stats]

        b_mse = np.array([s["base_mse_mean"]  for s in stats])
        f_mse = np.array([s["fix_mse_mean"]   for s in stats])
        g_mse = np.array([s["gated_mse_mean"] for s in stats])
        b_sem = np.array([s["base_mse_sem"]   for s in stats])
        f_sem = np.array([s["fix_mse_sem"]    for s in stats])
        g_sem = np.array([s["gated_mse_sem"]  for s in stats])

        b_cos = [s["base_cos_mean"]  for s in stats]
        f_cos = [s["fix_cos_mean"]   for s in stats]
        g_cos = [s["gated_cos_mean"] for s in stats]

        gain  = [s["gain_pct"]  for s in stats]
        cd    = [s["cohens_d"]  for s in stats]

        fig, axes = plt.subplots(2, 2, figsize=(14, 9), dpi=150)
        fig.suptitle(
            f"{env}: RAG-WM vs JEPA-WM Baseline  "
            f"(N={N} held-out trajectories, H={self.cfg.horizon})",
            fontsize=13, fontweight="bold",
        )

        # (a) MSE
        ax = axes[0, 0]
        ax.plot(steps, b_mse, label="JEPA-WM Baseline",         color=_COLORS["baseline"], marker="o", lw=2)
        ax.fill_between(steps, b_mse - b_sem, b_mse + b_sem,    color=_COLORS["baseline"], alpha=0.15)
        ax.plot(steps, f_mse, label=r"RAG-WM (fixed $\lambda$)", color=_COLORS["rag_fix"],  marker="s", lw=2)
        ax.fill_between(steps, f_mse - f_sem, f_mse + f_sem,    color=_COLORS["rag_fix"],  alpha=0.15)
        ax.plot(steps, g_mse, label="RAG-WM (adaptive gate)",   color=_COLORS["rag_gate"], marker="^", lw=2)
        ax.fill_between(steps, g_mse - g_sem, g_mse + g_sem,    color=_COLORS["rag_gate"], alpha=0.15)
        ax.set_title("(a) Latent Patch MSE  ↓"); ax.set_xlabel("Horizon H"); ax.set_ylabel("Mean MSE")
        ax.legend(fontsize=9); ax.set_xticks(steps)

        # (b) Cosine similarity
        ax = axes[0, 1]
        ax.plot(steps, b_cos, label="JEPA-WM Baseline",         color=_COLORS["baseline"], marker="o", lw=2)
        ax.plot(steps, f_cos, label=r"RAG-WM (fixed $\lambda$)", color=_COLORS["rag_fix"],  marker="s", lw=2)
        ax.plot(steps, g_cos, label="RAG-WM (adaptive gate)",   color=_COLORS["rag_gate"], marker="^", lw=2)
        ax.set_title("(b) Cosine Similarity  ↑"); ax.set_xlabel("Horizon H"); ax.set_ylabel("Cosine Sim")
        ax.legend(fontsize=9); ax.set_xticks(steps)

        # (c) Gain %
        ax = axes[1, 0]
        colors = [_COLORS["rag_fix"] if g >= 0 else _COLORS["baseline"] for g in gain]
        ax.bar(steps, gain, color=colors, alpha=0.8, edgecolor="white", width=0.7)
        ax.axhline(0, color="black", lw=0.8, ls="--")
        ax.set_title("(c) Relative MSE Gain vs Baseline (%)"); ax.set_xlabel("Horizon H"); ax.set_ylabel("Gain (%)")
        ax.set_xticks(steps)

        # (d) Cohen's d
        ax = axes[1, 1]
        ax.plot(steps, cd, color=_COLORS["rag_fix"], marker="D", lw=2, label="Cohen's d")
        for thresh, col, lbl in [(0.2, "grey", "small (0.2)"), (0.5, "orange", "medium (0.5)"), (0.8, "red", "large (0.8)")]:
            ax.axhline(thresh, ls="--", color=col, lw=0.8, label=lbl)
        ax.set_title("(d) Effect Size (Cohen's d)"); ax.set_xlabel("Horizon H"); ax.set_ylabel("Cohen's d")
        ax.legend(fontsize=8); ax.set_xticks(steps)

        path = self.cfg.output_dir / f"{env_slug}_benchmark_figure.png"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        print(f"[INFO] Saved figure to: {path}")
        return [path]

    # ── main pipeline ─────────────────────────────────────────────────────────

    def run(self) -> None:
        """Execute the full benchmark pipeline end-to-end."""
        cfg = self.cfg
        print("=" * 80)
        print(f"  BENCHMARK: JEPA-WM vs RAG-WM  |  Environment: {cfg.env_name}")
        print("=" * 80)

        model, preprocessor = self.load_model()

        agent = RAGWorldModelAgent(
            base_model=model,
            preprocessor=preprocessor,
            rag_lambda=cfg.rag_lambda,
            device=cfg.device,
        )

        memory_bank = self.build_memory_bank(agent)
        agent.memory_bank = memory_bank

        test_episodes = self.get_test_episodes()
        print(f"[INFO] {len(test_episodes)} candidate test episodes loaded.")

        all_results = []
        for ep in test_episodes:
            res = self.evaluate_episode(agent, ep)
            if res is not None:
                all_results.append(res)

        N = len(all_results)
        print(f"[INFO] Evaluated {N} valid episodes across H={cfg.horizon} steps.")

        if N == 0:
            print("[WARNING] No valid episodes found. Aborting.")
            return

        stats = self.aggregate_stats(all_results)
        self.print_table(stats)
        self.save_json(stats, extra={"num_test_trajectories": N})
        self.save_figures(stats, N=N)
