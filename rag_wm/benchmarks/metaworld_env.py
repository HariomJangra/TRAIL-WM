"""
metaworld_env.py
────────────────
MetaWorld benchmark adapter for RAG-WM.
Environment: 3-D articulated arm manipulation (multi-task).

Data layout:
    rag_wm/data/metaworld/data/*.parquet   — 50 shards
    rag_wm/models/jepa_wm_metaworld.pth.tar

Memory split:  shards  0..29  (30 shards, ≤80 episodes)
Test split:    shards 30..49  (20 shards, 2 episodes each)
"""

from __future__ import annotations

import glob
import io
import sys
from pathlib import Path
from typing import Any, Optional

import imageio.v2 as imageio
import numpy as np
import pyarrow.parquet as pq
import torch
import yaml

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
JEPA_ROOT = WORKSPACE_ROOT.parent / "jepa-wms"
for _p in (str(WORKSPACE_ROOT), str(JEPA_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from app.plan_common.datasets import get_data_stats
from app.plan_common.datasets.preprocessor import Preprocessor
from app.plan_common.datasets.transforms import make_inverse_transforms, make_transforms
from app.vjepa_wm.modelcustom.simu_env_planning.vit_enc_preds import init_module
from benchmarks.benchmark_base import BenchmarkConfig, EnvironmentBenchmark
from memory.episodic_memory import EpisodicMemoryBank
from agent.rag_agent import RAGWorldModelAgent
from src.utils.yaml_utils import expand_env_vars


class MetaWorldBenchmark(EnvironmentBenchmark):
    """RAG-WM benchmark for the MetaWorld multi-task manipulation suite."""

    # ── data paths ────────────────────────────────────────────────────────────
    _CHECKPOINT = "models/jepa_wm_metaworld.pth.tar"
    _CONFIG     = (
        "configs/evals/simu_env_planning/mw/"
        "jepa-wm/reach-wall_L2_cem_sourcexp_H6_nas3_ctxt2_r256_alpha0.1_ep48_decode.yaml"
    )
    _DATA_DIR   = "data/metaworld/data"

    def load_model(self) -> tuple[Any, Any]:
        """Load MetaWorld JEPA-WM from YAML config + checkpoint."""
        config_path = JEPA_ROOT / self._CONFIG
        with open(config_path, "r") as f:
            args_eval = yaml.safe_load(f)
        args_eval = expand_env_vars(args_eval)

        mk  = args_eval["model_kwargs"]
        cfgs_data     = mk.get("data", {})
        cfgs_data_aug = mk.get("data_aug", {})
        wrapper_kw    = mk.get("wrapper_kwargs", {})
        pretrain_kw   = mk.get("pretrain_kwargs", {})

        # strip head configs — not needed for eval
        for d in (mk, pretrain_kw):
            d.pop("heads_cfg", None)

        data_stats = get_data_stats("metaworld")
        img_size   = cfgs_data.get("img_size", 224)

        transform = make_transforms(
            img_size=img_size,
            normalize=cfgs_data_aug.get("normalize", [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]]),
            random_horizontal_flip=False,
            random_resize_aspect_ratio=(1.0, 1.0),
            random_resize_scale=(1.0, 1.0),
            reprob=0.0, auto_augment=False, motion_shift=False,
        )
        inverse_transform = make_inverse_transforms(img_size=img_size, **cfgs_data_aug)

        preprocessor = Preprocessor(
            action_mean=torch.tensor(data_stats["action_mean"]),
            action_std= torch.tensor(data_stats["action_std"]),
            state_mean= torch.tensor(data_stats["state_mean"]),
            state_std=  torch.tensor(data_stats["state_std"]),
            proprio_mean=torch.tensor(data_stats["proprio_mean"]),
            proprio_std= torch.tensor(data_stats["proprio_std"]),
            transform=transform,
            inverse_transform=inverse_transform,
        )

        ckpt_path = WORKSPACE_ROOT / self._CHECKPOINT
        model = init_module(
            folder=str(ckpt_path.parent),
            checkpoint=ckpt_path.name,
            module_name=mk.get("module_name"),
            model_kwargs=pretrain_kw,
            wrapper_kwargs=wrapper_kw,
            cfgs_data=cfgs_data,
            device=torch.device(self.cfg.device),
            action_dim=data_stats["action_dim"],
            proprio_dim=data_stats["proprio_dim"],
            preprocessor=preprocessor,
            heads_cfg={},
        )
        model.eval()
        return model, preprocessor

    def build_memory_bank(self, agent: RAGWorldModelAgent) -> EpisodicMemoryBank:
        """Use the parquet-based high-level builder (shards 0..29)."""
        all_shards    = sorted(glob.glob(str(WORKSPACE_ROOT / self._DATA_DIR / "*.parquet")))
        memory_shards = all_shards[:30]
        print(f"[INFO] Building MetaWorld memory bank from {len(memory_shards)} shards...")

        bank = EpisodicMemoryBank(
            device=self.cfg.device,
            top_k=self.cfg.top_k,
            temperature=self.cfg.temperature,
        )
        bank.build_from_parquet_files(
            parquet_paths=memory_shards,
            model=agent.model,
            preprocessor=agent.preprocessor,
            max_episodes_per_file=3,
            max_total_episodes=80,
        )
        return bank

    def get_test_episodes(self) -> list[dict]:
        """Load test episodes from held-out shards 30..49."""
        all_shards  = sorted(glob.glob(str(WORKSPACE_ROOT / self._DATA_DIR / "*.parquet")))
        test_shards = all_shards[30:50]
        min_frames  = self.cfg.ctxt_window + self.cfg.horizon + 1

        episodes = []
        for shard_path in test_shards:
            try:
                table = pq.read_table(shard_path)
            except Exception:
                continue
            for ep_idx in range(min(2, len(table))):
                ep = self._load_parquet_episode(table, ep_idx)
                if ep is not None and len(ep["frames"]) >= min_frames:
                    episodes.append(ep)
        return episodes

    # ── internal helpers ──────────────────────────────────────────────────────

    def _load_parquet_episode(self, table, ep_idx: int) -> Optional[dict]:
        try:
            task_name  = table["task"][ep_idx].as_py()
            vid_bytes  = table["video"][ep_idx].as_py()["bytes"]
            reader     = imageio.get_reader(io.BytesIO(vid_bytes), format="mp4")
            frames     = np.stack([f for f in reader]); reader.close()
            frames     = np.transpose(frames, (0, 3, 1, 2))   # (T, C, H, W)
            actions    = np.array(table["actions"][ep_idx].as_py(), dtype=np.float32)
            states     = np.array(table["states"][ep_idx].as_py(),  dtype=np.float32)
            T          = min(len(frames), len(actions), len(states))
            return {
                "task":    task_name,
                "frames":  torch.from_numpy(frames[:T]).float(),
                "actions": torch.from_numpy(actions[:T]).float(),
                "proprios": torch.from_numpy(states[:T, :4]).float(),
            }
        except Exception:
            return None

    def preprocess_episode(self, ep_data: dict, agent: RAGWorldModelAgent) -> Optional[dict]:
        cfg   = self.cfg
        total = cfg.ctxt_window + cfg.horizon
        if len(ep_data["frames"]) < total or len(ep_data["actions"]) < total - 1:
            return None

        frames   = ep_data["frames"][:total]        # (T, C, H, W) raw [0..255] float
        proprios = ep_data["proprios"][:total]      # (T, 4) raw proprio
        actions  = ep_data["actions"][cfg.ctxt_window - 1: cfg.ctxt_window - 1 + cfg.horizon]

        # MetaWorld model.encode() expects raw pixel values [0..255] and raw proprio;
        # normalization (visual / 255.0 + preprocessor.transform, and normalize_proprios)
        # is performed internally by the model encoder.
        frames_t   = frames.unsqueeze(0).to(cfg.device)
        prop_t     = proprios.unsqueeze(0).to(cfg.device)
        act_norm   = agent.preprocessor.normalize_actions(actions).to(cfg.device)
        act_suffix = act_norm.unsqueeze(1)          # (H, 1, action_dim)

        return {
            "obs_dict":   {"visual": frames_t, "proprio": prop_t},
            "act_suffix": act_suffix,
        }

    # ── extended run: per-task breakdown ─────────────────────────────────────

    def run(self) -> None:
        """Run base pipeline and add a per-task breakdown to the JSON."""
        cfg = self.cfg
        print("=" * 80)
        print(f"  BENCHMARK: JEPA-WM vs RAG-WM  |  Environment: {cfg.env_name}")
        print("=" * 80)

        model, preprocessor = self.load_model()
        agent = RAGWorldModelAgent(
            base_model=model, preprocessor=preprocessor,
            rag_lambda=cfg.rag_lambda, device=cfg.device,
        )
        agent.memory_bank = self.build_memory_bank(agent)

        test_episodes = self.get_test_episodes()
        print(f"[INFO] {len(test_episodes)} candidate test episodes loaded.")

        # Store episodes alongside their results for per-task aggregation
        episode_results: list[tuple[str, list[dict]]] = []
        for ep in test_episodes:
            res = self.evaluate_episode(agent, ep)
            if res is not None:
                task = ep.get("task", "unknown")
                episode_results.append((task, res))

        N = len(episode_results)
        print(f"[INFO] Evaluated {N} valid episodes across H={cfg.horizon} steps.")
        if N == 0:
            print("[WARNING] No valid episodes found. Aborting."); return

        all_results = [r for _, r in episode_results]
        stats       = self.aggregate_stats(all_results)
        self.print_table(stats)

        # Per-task breakdown
        per_task: dict = {}
        all_tasks = sorted(set(t for t, _ in episode_results))
        for task_name in all_tasks:
            t_res  = [r for t, r in episode_results if t == task_name]
            t_base = float(np.mean([[s["mse_base"] for s in r] for r in t_res]))
            t_fix  = float(np.mean([[s["mse_fix"]  for s in r] for r in t_res]))
            per_task[task_name] = {
                "n_episodes":    len(t_res),
                "baseline_mean": t_base,
                "rag_mean":      t_fix,
                "gain_pct":      (t_base - t_fix) / t_base * 100.0,
            }

        self.save_json(
            stats,
            extra={
                "num_test_trajectories": N,
                "memory_transitions":    agent.memory_bank.num_transitions,
                "per_task_summary":      per_task,
            },
        )
        self.save_figures(stats, N=N)


if __name__ == "__main__":
    cfg = BenchmarkConfig(env_name="MetaWorld")
    MetaWorldBenchmark(cfg).run()
