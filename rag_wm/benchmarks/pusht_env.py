"""
pusht_env.py
────────────
Push-T benchmark adapter for RAG-WM.
Environment: Continuous 2-D planar block pushing.

Data layout:
    rag_wm/data/pusht_noise/
        train/
            obses/episode_NNN.mp4
            rel_actions.pth     — (N_eps, T, 2)  ×100 scale
            states.pth          — (N_eps, T, 2)
            velocities.pth      — (N_eps, T, 2)
        val/
            obses/episode_NNN.mp4
            (same .pth files)
    rag_wm/models/jepa_wm_pusht.pth.tar

Memory: all episodes in train/
Test:   all episodes in val/
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Optional

if sys.platform == "win32":
    if sys.stdout and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if sys.stderr and hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

import imageio.v2 as imageio
import numpy as np
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


class PushTBenchmark(EnvironmentBenchmark):
    """RAG-WM benchmark for Push-T continuous planar manipulation."""

    _CHECKPOINT = "models/jepa_wm_pusht.pth.tar"
    _CONFIG = (
        "configs/evals/simu_env_planning/pt/"
        "jepa-wm/pt_L2_cem_sourcedset_H6_nas6_ctxt2_r224_alpha0.1_ep96_decode.yaml"
    )
    _TRAIN_DIR = "data/pusht_noise/train"
    _VAL_DIR   = "data/pusht_noise/val"
    _N_MEMORY_EPS = 50

    def load_model(self) -> tuple[Any, Any]:
        """Load Push-T JEPA-WM from YAML config + checkpoint."""
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

        data_stats = get_data_stats("pusht")
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
        """Encode all train episodes and index transitions."""
        train_dir = WORKSPACE_ROOT / self._TRAIN_DIR
        ep_paths  = sorted((train_dir / "obses").glob("episode_*.mp4"))
        if self._N_MEMORY_EPS is not None:
            ep_paths = ep_paths[: self._N_MEMORY_EPS]
        bank      = EpisodicMemoryBank(
            device=self.cfg.device,
            top_k=self.cfg.top_k,
            temperature=self.cfg.temperature,
        )

        print(f"[INFO] Building Push-T memory bank from {len(ep_paths)} train episodes...")
        t0 = time.time()

        all_keys, all_vis, all_prop, all_act = [], [], [], []
        all_tgt_vis, all_tgt_prop, all_delta = [], [], []
        all_tasks = []

        with torch.no_grad():
            for ep_path in ep_paths:
                ep_idx = int(ep_path.stem.split("_")[-1])
                ep = self._load_raw_episode(str(train_dir), ep_idx)
                if ep is None or len(ep["frames"]) < 5:
                    continue

                frames   = ep["frames"]
                actions  = ep["actions"]
                proprios = ep["proprios"]
                T        = len(frames)

                frames_t  = agent.preprocessor.transform(frames / 255.0).to(self.cfg.device)
                prop_norm = agent.preprocessor.normalize_proprios(proprios).to(self.cfg.device)
                act_norm  = agent.preprocessor.normalize_actions(actions).to(self.cfg.device)

                latents = agent.model.encode({
                    "visual": frames_t.unsqueeze(0),
                    "proprio": prop_norm.unsqueeze(0),
                })
                vis_l  = latents["visual"][0]
                prop_l = latents["proprio"][0]

                n = T - 1
                curr_vis, next_vis   = vis_l[:n], vis_l[1: n + 1]
                curr_prop, next_prop = prop_l[:n], prop_l[1: n + 1]
                deltas               = next_vis - curr_vis

                vis_g = F.normalize(curr_vis.squeeze(1).mean(dim=(1, 2)), p=2, dim=-1)
                act_g = F.normalize(act_norm[:n], p=2, dim=-1)
                keys  = F.normalize(torch.cat([vis_g, act_g], dim=-1), p=2, dim=-1)

                all_keys.append(keys.cpu());      all_vis.append(curr_vis.cpu())
                all_prop.append(curr_prop.cpu()); all_act.append(act_norm[:n].cpu())
                all_tgt_vis.append(next_vis.cpu());  all_tgt_prop.append(next_prop.cpu())
                all_delta.append(deltas.cpu());   all_tasks.extend(["pusht"] * n)

        bank.query_keys      = torch.cat(all_keys).to(self.cfg.device)
        bank.memory_visual   = torch.cat(all_vis).to(self.cfg.device)
        bank.memory_proprio  = torch.cat(all_prop).to(self.cfg.device)
        bank.memory_action   = torch.cat(all_act).to(self.cfg.device)
        bank.target_visual   = torch.cat(all_tgt_vis).to(self.cfg.device)
        bank.target_proprio  = torch.cat(all_tgt_prop).to(self.cfg.device)
        bank.delta_visual    = torch.cat(all_delta).to(self.cfg.device)
        bank.source_tasks    = all_tasks
        bank.num_transitions = len(bank.query_keys)
        bank.tasks_indexed   = {"pusht"}

        print(f"[INFO] Push-T memory bank: {bank.num_transitions} transitions in {time.time() - t0:.1f}s")
        return bank

    def get_test_episodes(self) -> list[dict]:
        """Load all held-out validation episodes."""
        val_dir  = WORKSPACE_ROOT / self._VAL_DIR
        ep_paths = sorted((val_dir / "obses").glob("episode_*.mp4"))
        min_len  = self.cfg.ctxt_window + self.cfg.horizon + 1

        episodes = []
        for ep_path in ep_paths:
            ep_idx = int(ep_path.stem.split("_")[-1])
            ep = self._load_raw_episode(str(val_dir), ep_idx)
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

    def _get_tensors(self, data_dir: str):
        if not hasattr(self, "_tensors_cache"):
            self._tensors_cache = {}
        if data_dir not in self._tensors_cache:
            d = Path(data_dir)
            rel_actions = torch.load(d / "rel_actions.pth", map_location="cpu", mmap=True)
            states      = torch.load(d / "states.pth",      map_location="cpu", mmap=True)
            velocities  = torch.load(d / "velocities.pth",  map_location="cpu", mmap=True)
            self._tensors_cache[data_dir] = (rel_actions, states, velocities)
        return self._tensors_cache[data_dir]

    def _load_raw_episode(self, data_dir: str, ep_idx: int) -> Optional[dict]:
        """Load a single Push-T episode from MP4 + .pth tensors."""
        vid_file = Path(data_dir) / "obses" / f"episode_{ep_idx:03d}.mp4"
        if not vid_file.exists():
            return None

        reader = imageio.get_reader(str(vid_file), format="mp4")
        frames = np.stack([f for f in reader]); reader.close()
        frames = torch.from_numpy(np.transpose(frames, (0, 3, 1, 2))).float()   # (T, C, H, W)

        rel_actions, states, velocities = self._get_tensors(data_dir)

        actions  = rel_actions[ep_idx].float() / 100.0           # (T, 2)
        proprios = torch.cat([states[ep_idx, :, :2], velocities[ep_idx]], dim=-1).float()  # (T, 4)

        T = min(len(frames), len(actions), len(proprios))
        return {
            "frames":   frames[:T],
            "actions":  actions[:T],
            "proprios": proprios[:T],
        }


if __name__ == "__main__":
    cfg = BenchmarkConfig(env_name="Push-T")
    PushTBenchmark(cfg).run()
