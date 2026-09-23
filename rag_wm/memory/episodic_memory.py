"""
Dense Episodic Memory Bank for Joint-Embedding World Models (JEPA-WM).
- Indexes transitions (s_t, a_t -> s_{t+1}, delta_s) in DINO latent space.
- GPU-accelerated cosine similarity search and softmax temperature weighting.
- Supports nearest-neighbor distance auditing, memory scaling, and K=0 baseline passthrough.
"""

import io
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import imageio.v2 as imageio
import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from einops import rearrange
from tensordict import TensorDict


class EpisodicMemoryBank:
    def __init__(
        self,
        device: str = "cuda:0",
        top_k: int = 3,
        temperature: float = 0.1,
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.top_k = top_k
        self.temperature = temperature

        # Core memory storage tensors on GPU
        self.query_keys: Optional[torch.Tensor] = None      # (N, key_dim)
        self.memory_visual: Optional[torch.Tensor] = None   # (N, 1, 16, 16, 384)
        self.memory_proprio: Optional[torch.Tensor] = None  # (N, 1, P_tok, 16)
        self.memory_action: Optional[torch.Tensor] = None   # (N, action_dim)
        self.target_visual: Optional[torch.Tensor] = None   # (N, 1, 16, 16, 384)
        self.target_proprio: Optional[torch.Tensor] = None  # (N, 1, P_tok, 16)
        self.delta_visual: Optional[torch.Tensor] = None    # (N, 1, 16, 16, 384)
        self.source_tasks: List[str] = []

        self.num_transitions = 0
        self.tasks_indexed = set()

    def build_from_parquet_files(
        self,
        parquet_paths: List[Union[str, Path]],
        model,
        preprocessor,
        max_episodes_per_file: int = 3,
        max_total_episodes: int = 80,
    ):
        """Extracts dense patch representations and builds the GPU vector index."""
        all_keys = []
        all_mem_vis = []
        all_mem_prop = []
        all_mem_act = []
        all_tgt_vis = []
        all_tgt_prop = []
        all_delta_vis = []
        all_tasks = []

        total_episodes = 0
        for p_path in parquet_paths:
            if total_episodes >= max_total_episodes:
                break
            try:
                table = pq.read_table(str(p_path))
            except Exception:
                continue

            num_episodes_in_file = min(len(table), max_episodes_per_file)

            for ep_idx in range(num_episodes_in_file):
                if total_episodes >= max_total_episodes:
                    break

                task_name = table["task"][ep_idx].as_py()
                self.tasks_indexed.add(task_name)

                video_bytes = table["video"][ep_idx].as_py()["bytes"]
                reader = imageio.get_reader(io.BytesIO(video_bytes), format="mp4")
                frames = np.stack([frame for frame in reader])
                reader.close()
                frames = np.transpose(frames, (0, 3, 1, 2))  # (T, C, H, W)

                actions = np.array(table["actions"][ep_idx].as_py(), dtype=np.float32)
                states = np.array(table["states"][ep_idx].as_py(), dtype=np.float32)

                T_len = min(len(frames), len(actions), len(states))
                if T_len < 3:
                    continue

                frames_tensor = torch.from_numpy(frames[:T_len]).float().unsqueeze(0).to(self.device)
                states_tensor = torch.from_numpy(states[:T_len, :4]).float().unsqueeze(0).to(self.device)
                actions_tensor = torch.from_numpy(actions[: T_len - 1]).float()
                actions_norm = preprocessor.normalize_actions(actions_tensor).to(self.device)

                with torch.no_grad():
                    obs_dict = {"visual": frames_tensor, "proprio": states_tensor}
                    encoded = model.encode(obs_dict)
                    vis_latents = encoded["visual"][0]    # (T, 1, 16, 16, 384)
                    prop_latents = encoded["proprio"][0]  # (T, 1, P_tok, 16)

                    num_transitions_ep = T_len - 1
                    curr_vis = vis_latents[:num_transitions_ep]
                    next_vis = vis_latents[1:num_transitions_ep + 1]
                    curr_prop = prop_latents[:num_transitions_ep]
                    next_prop = prop_latents[1:num_transitions_ep + 1]
                    deltas = next_vis - curr_vis

                    # Compact key: pooled spatial latent + normalized action
                    vis_global = curr_vis.squeeze(1).mean(dim=(1, 2))  # (T-1, 384)
                    vis_global_norm = F.normalize(vis_global, p=2, dim=-1)
                    act_norm = F.normalize(actions_norm, p=2, dim=-1)
                    key_descriptor = torch.cat([vis_global_norm, act_norm], dim=-1)
                    key_descriptor = F.normalize(key_descriptor, p=2, dim=-1)

                    all_keys.append(key_descriptor.cpu())
                    all_mem_vis.append(curr_vis.cpu())
                    all_mem_prop.append(curr_prop.cpu())
                    all_mem_act.append(actions_norm.cpu())
                    all_tgt_vis.append(next_vis.cpu())
                    all_tgt_prop.append(next_prop.cpu())
                    all_delta_vis.append(deltas.cpu())
                    all_tasks.extend([task_name] * num_transitions_ep)

                total_episodes += 1

        self.query_keys = torch.cat(all_keys, dim=0).to(self.device)
        self.memory_visual = torch.cat(all_mem_vis, dim=0).to(self.device)
        self.memory_proprio = torch.cat(all_mem_prop, dim=0).to(self.device)
        self.memory_action = torch.cat(all_mem_act, dim=0).to(self.device)
        self.target_visual = torch.cat(all_tgt_vis, dim=0).to(self.device)
        self.target_proprio = torch.cat(all_tgt_prop, dim=0).to(self.device)
        self.delta_visual = torch.cat(all_delta_vis, dim=0).to(self.device)
        self.source_tasks = all_tasks

        self.num_transitions = len(self.query_keys)
        print(f" [✓] Episodic Memory Built: {self.num_transitions} transitions indexed across {total_episodes} episodes ({len(self.tasks_indexed)} tasks).")

    def get_sliced_subview(self, max_size: int):
        """Returns a temporary memory bank sliced to max_size for capacity scaling ablation."""
        sub_bank = EpisodicMemoryBank(device=str(self.device), top_k=self.top_k, temperature=self.temperature)
        sub_size = min(max_size, self.num_transitions)
        sub_bank.query_keys = self.query_keys[:sub_size]
        sub_bank.memory_visual = self.memory_visual[:sub_size]
        sub_bank.memory_proprio = self.memory_proprio[:sub_size]
        sub_bank.memory_action = self.memory_action[:sub_size]
        sub_bank.target_visual = self.target_visual[:sub_size]
        sub_bank.target_proprio = self.target_proprio[:sub_size]
        sub_bank.delta_visual = self.delta_visual[:sub_size]
        sub_bank.source_tasks = self.source_tasks[:sub_size]
        sub_bank.num_transitions = sub_size
        sub_bank.tasks_indexed = set(self.source_tasks[:sub_size])
        return sub_bank

    @torch.no_grad()
    def retrieve(
        self,
        query_visual: torch.Tensor,
        query_action: torch.Tensor,
        top_k: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Retrieves top-K most similar transitions for a given query (s_t, a_t).
        """
        k = top_k if top_k is not None else self.top_k
        temp = temperature if temperature is not None else self.temperature

        # Sanity check: K=0 returns zero delta (pure baseline)
        if k == 0 or self.num_transitions == 0:
            B = query_visual.shape[0]
            dummy_delta = torch.zeros(B, 1, 16, 16, 384, device=self.device)
            return {
                "topk_indices": torch.zeros(B, 1, dtype=torch.long, device=self.device),
                "topk_scores": torch.zeros(B, 1, device=self.device),
                "topk_weights": torch.ones(B, 1, device=self.device),
                "weighted_delta": dummy_delta,
                "top1_similarity": torch.zeros(B, device=self.device),
            }

        k = min(k, self.num_transitions)

        # Build query descriptor
        if query_visual.dim() == 5:
            q_vis_global = query_visual.squeeze(1).mean(dim=(1, 2))  # (B, 384)
        else:
            q_vis_global = query_visual.mean(dim=(1, 2))             # (B, 384)
            
        q_vis_norm = F.normalize(q_vis_global, p=2, dim=-1)
        
        mem_act_dim = self.memory_action.shape[-1]
        if query_action.shape[-1] > mem_act_dim:
            q_act = query_action[:, :mem_act_dim]
        elif query_action.shape[-1] < mem_act_dim:
            q_act = F.pad(query_action, (0, mem_act_dim - query_action.shape[-1]))
        else:
            q_act = query_action

        q_act_norm = F.normalize(q_act, p=2, dim=-1)
        query_key = torch.cat([q_vis_norm, q_act_norm], dim=-1)
        query_key = F.normalize(query_key, p=2, dim=-1)

        # Cosine similarity matching: (B, key_dim) @ (N, key_dim)^T -> (B, N)
        similarity = torch.matmul(query_key, self.query_keys.T)
        topk_scores, topk_indices = torch.topk(similarity, k=k, dim=-1)  # (B, k)

        # Softmax weighting
        weights = F.softmax(topk_scores / temp, dim=-1)  # (B, k)

        B = query_visual.shape[0]
        retrieved_deltas = self.delta_visual[topk_indices]  # (B, k, 1, 16, 16, 384)
        weights_expanded = weights.view(B, k, 1, 1, 1, 1)
        weighted_delta = torch.sum(weights_expanded * retrieved_deltas, dim=1)  # (B, 1, 16, 16, 384)

        return {
            "topk_indices": topk_indices,
            "topk_scores": topk_scores,
            "topk_weights": weights,
            "weighted_delta": weighted_delta,
            "top1_similarity": topk_scores[:, 0],  # (B,)
        }

    @torch.no_grad()
    def audit_nearest_neighbor_distances(
        self,
        query_visual: torch.Tensor,
        query_action: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Audits Euclidean and Cosine distances to the top-1 retrieved neighbor to verify non-trivial retrieval.
        """
        ret = self.retrieve(query_visual, query_action, top_k=1)
        top1_sim = ret["top1_similarity"]  # Cosine similarity
        top1_idx = ret["topk_indices"][:, 0]

        # Compute full spatial latent Euclidean distance
        if query_visual.dim() == 5:
            q_vis = query_visual
        else:
            q_vis = query_visual.unsqueeze(1)
        
        mem_vis = self.memory_visual[top1_idx]
        euc_dist = torch.norm(q_vis - mem_vis, p=2, dim=(-1, -2, -3)).mean().item()

        return {
            "top1_cosine_sim": top1_sim.mean().item(),
            "top1_euclidean_dist": euc_dist,
        }
