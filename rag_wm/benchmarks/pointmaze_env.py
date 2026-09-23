"""
pointmaze_env.py
────────────────
PointMaze benchmark adapter for RAG-WM.
Environment: 2-D discrete obstacle maze navigation.

Data layout:
    rag_wm/data/point_maze/point_maze/
        obses/episode_NNN.pth   — (T, 224, 224, 3) uint8
        states.pth              — (N_eps, T, 4)
        actions.pth             — (N_eps, T, 2)
        seq_lengths.pth         — (N_eps,)
    rag_wm/models/jepa_wm_pointmaze.pth.tar

Memory split: episodes  0..39  (40 episodes)
Test split:   episodes 40..64  (25 episodes)
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn.functional as F
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


class PointMazeBenchmark(EnvironmentBenchmark):
    """RAG-WM benchmark for PointMaze discrete maze navigation."""

    _CHECKPOINT = "models/jepa_wm_pointmaze.pth.tar"
    _CONFIG = (
        "configs/evals/simu_env_planning/mz/"
        "jepa-wm/mz_L2_cem_sourcerandstate_H6_nas6_ctxt2_r224_alpha0.1_ep96_decode.yaml"
    )
    _DATA_DIR      = "data/point_maze/point_maze"
    _N_MEMORY_EPS  = 40    # episodes used for memory bank
    _TEST_EP_START = 40
    _TEST_EP_END   = 65

    def load_model(self) -> tuple[Any, Any]:
        """Load PointMaze JEPA-WM from YAML config + checkpoint."""
        config_path = JEPA_ROOT / self._CONFIG
        with open(config_path, "r") as f:
            args_eval = yaml.safe_load(f)
        args_eval = expand_env_vars(args_eval)

        mk            = args_eval["model_kwargs"]
        cfgs_data     = mk.get("data", {})
        cfgs_data_aug = mk.get("data_aug", {})
        wrapper_kw    = mk.get("wrapper_kwargs", {})
        pretrain_kw   = mk.get("pretrain_kwargs", {})
        for d in (mk, pretrain_kw):
            d.pop("heads_cfg", None)

        data_stats = get_data_stats("pointmaze")
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
        """Manually encode episodes 0..N_MEMORY_EPS and index transitions."""
        data_dir = WORKSPACE_ROOT / self._DATA_DIR
        bank     = EpisodicMemoryBank(
            device=self.cfg.device,
            top_k=self.cfg.top_k,
            temperature=self.cfg.temperature,
        )

        print(f"[INFO] Building PointMaze memory bank from episodes 0..{self._N_MEMORY_EPS - 1}...")
        t0 = time.time()

        all_keys, all_vis, all_prop, all_act = [], [], [], []
        all_tgt_vis, all_tgt_prop, all_delta = [], [], []
        all_tasks = []

        with torch.no_grad():
            for ep_idx in range(self._N_MEMORY_EPS):
                ep = self._load_raw_episode(data_dir, ep_idx)
                if ep is None or len(ep["frames"]) < 5:
                    continue

                frames   = ep["frames"]
                actions  = ep["actions"]
                proprios = ep["proprios"]
                T        = len(frames)

                frames_t   = agent.preprocessor.transform(frames / 255.0).to(self.cfg.device)
                prop_norm  = agent.preprocessor.normalize_proprios(proprios).to(self.cfg.device)
                act_norm   = agent.preprocessor.normalize_actions(actions).to(self.cfg.device)

                latents = agent.model.encode({
                    "visual": frames_t.unsqueeze(0),
                    "proprio": prop_norm.unsqueeze(0),
                })
                vis_l  = latents["visual"][0]    # (T, 1, 16, 16, D)
                prop_l = latents["proprio"][0]   # (T, 1, P, D)

                n = T - 1
                curr_vis, next_vis   = vis_l[:n], vis_l[1: n + 1]
                curr_prop, next_prop = prop_l[:n], prop_l[1: n + 1]
                deltas               = next_vis - curr_vis

                vis_g  = F.normalize(curr_vis.squeeze(1).mean(dim=(1, 2)), p=2, dim=-1)
                act_g  = F.normalize(act_norm[:n], p=2, dim=-1)
                keys   = F.normalize(torch.cat([vis_g, act_g], dim=-1), p=2, dim=-1)

                all_keys.append(keys.cpu());      all_vis.append(curr_vis.cpu())
                all_prop.append(curr_prop.cpu()); all_act.append(act_norm[:n].cpu())
                all_tgt_vis.append(next_vis.cpu());  all_tgt_prop.append(next_prop.cpu())
                all_delta.append(deltas.cpu());   all_tasks.extend(["pointmaze"] * n)

        bank.query_keys      = torch.cat(all_keys).to(self.cfg.device)
        bank.memory_visual   = torch.cat(all_vis).to(self.cfg.device)
        bank.memory_proprio  = torch.cat(all_prop).to(self.cfg.device)
        bank.memory_action   = torch.cat(all_act).to(self.cfg.device)
        bank.target_visual   = torch.cat(all_tgt_vis).to(self.cfg.device)
        bank.target_proprio  = torch.cat(all_tgt_prop).to(self.cfg.device)
        bank.delta_visual    = torch.cat(all_delta).to(self.cfg.device)
        bank.source_tasks    = all_tasks
        bank.num_transitions = len(bank.query_keys)
        bank.tasks_indexed   = {"pointmaze"}

        print(f"[INFO] PointMaze memory bank: {bank.num_transitions} transitions in {time.time() - t0:.1f}s")
        return bank

    def get_test_episodes(self) -> list[dict]:
        """Load held-out episodes TEST_EP_START..TEST_EP_END."""
        data_dir  = WORKSPACE_ROOT / self._DATA_DIR
        min_len   = self.cfg.ctxt_window + self.cfg.horizon + 1
        episodes  = []
        for ep_idx in range(self._TEST_EP_START, self._TEST_EP_END):
            ep = self._load_raw_episode(data_dir, ep_idx)
            if ep is not None and len(ep["frames"]) >= min_len:
                episodes.append(ep)
        return episodes

    def preprocess_episode(self, ep_data: dict, agent: RAGWorldModelAgent) -> Optional[dict]:
        cfg   = self.cfg
        total = cfg.ctxt_window + cfg.horizon
        if len(ep_data["frames"]) < total:
            return None

        frames   = ep_data["frames"][:total]
        proprios = ep_data["proprios"][:total]
        actions  = ep_data["actions"][cfg.ctxt_window - 1: cfg.ctxt_window - 1 + cfg.horizon]

        frames_t  = agent.preprocessor.transform(frames / 255.0).unsqueeze(0).to(cfg.device)
        prop_norm = agent.preprocessor.normalize_proprios(proprios).unsqueeze(0).to(cfg.device)
        act_norm  = agent.preprocessor.normalize_actions(actions).to(cfg.device)

        return {
            "obs_dict":   {"visual": frames_t, "proprio": prop_norm},
            "act_suffix": act_norm.unsqueeze(1),   # (H, 1, action_dim)
        }

    # ── internal ──────────────────────────────────────────────────────────────

    def _load_raw_episode(self, data_dir: Path, ep_idx: int) -> Optional[dict]:
        """Load frames, actions, proprios for a single episode index."""
        for fmt in (f"episode_{ep_idx:03d}.pth", f"episode_{ep_idx:04d}.pth"):
            obs_file = data_dir / "obses" / fmt
            if obs_file.exists():
                break
        else:
            return None

        frames_raw = torch.load(obs_file)          # (T, H, W, C) uint8
        frames     = frames_raw.permute(0, 3, 1, 2).float()   # (T, C, H, W)

        states_all  = torch.load(data_dir / "states.pth")     # (N, T, 4)
        actions_all = torch.load(data_dir / "actions.pth")    # (N, T, 2)
        seq_lengths = torch.load(data_dir / "seq_lengths.pth")

        T = min(int(seq_lengths[ep_idx]), len(frames))
        return {
            "frames":   frames[:T],
            "actions":  actions_all[ep_idx, :T].float(),
            "proprios": states_all[ep_idx,  :T].float(),
        }


if __name__ == "__main__":
    cfg = BenchmarkConfig(env_name="PointMaze")
    PointMazeBenchmark(cfg).run()
