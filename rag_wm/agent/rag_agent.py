"""
RAG-Augmented JEPA World Model Agent.
Integrates parametric JEPA Transformer dynamics with non-parametric episodic retrieval.
Supports Adaptive Confidence Gating, horizon decay schedules, and static residual baselines.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from tensordict import TensorDict

_RAG_WM_ROOT = Path(__file__).resolve().parent.parent
if str(_RAG_WM_ROOT) not in sys.path:
    sys.path.insert(0, str(_RAG_WM_ROOT))

from memory.episodic_memory import EpisodicMemoryBank


class RAGWorldModelAgent:
    def __init__(
        self,
        base_model,
        preprocessor,
        memory_bank: Optional[EpisodicMemoryBank] = None,
        rag_lambda: float = 0.30,
        device: str = "cuda:0",
    ):
        self.model = base_model
        self.preprocessor = preprocessor
        self.memory_bank = memory_bank
        self.rag_lambda = rag_lambda
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model.eval()

        # Mean empirical momentum residual (for static baseline comparison)
        self.static_momentum_delta: Optional[torch.Tensor] = None

    def compute_static_momentum(self):
        """Calculates global mean transition delta across memory as a non-retrieval control baseline."""
        if self.memory_bank is not None and self.memory_bank.num_transitions > 0:
            self.static_momentum_delta = self.memory_bank.delta_visual.mean(dim=0, keepdim=True)

    @torch.no_grad()
    def rollout_baseline(
        self,
        z_ctxt: TensorDict,
        act_suffix: torch.Tensor,
    ) -> torch.Tensor:
        """Standard unaugmented JEPA rollout (Pure Parametric Dynamics, K=0 / lambda=0)."""
        rollout_td = self.model.unroll(z_ctxt, act_suffix)
        pred_visual = rollout_td["visual"]  # (T_total, B, 1, 16, 16, 384)
        pred_patches = rearrange(pred_visual, "t b v h w d -> b t (v h w) d")
        return pred_patches

    @torch.no_grad()
    def rollout_static_residual(
        self,
        z_ctxt: TensorDict,
        act_suffix: torch.Tensor,
        rag_lambda: float = 0.30,
    ) -> torch.Tensor:
        """Control Baseline: Augments unroll with a constant global mean residual (no instance retrieval)."""
        if self.static_momentum_delta is None:
            self.compute_static_momentum()

        T, B, A = act_suffix.shape
        ctxt_window = z_ctxt["visual"].shape[1]
        vid_feats = z_ctxt["visual"].clone()
        prop_feats = z_ctxt["proprio"].clone()

        act_suffix_rearranged = rearrange(act_suffix, "t b ... -> b t ...")
        act_feats_suffix = self.model.model.encode_act(act_suffix_rearranged)

        for h in range(T):
            new_act_feat = act_feats_suffix[:, h : h + 1]
            act_feats = new_act_feat if h == 0 else torch.cat([act_feats, new_act_feat], dim=1)

            pred_vid, _, pred_prop = self.model.model.forward_pred(
                vid_feats[:, -ctxt_window:],
                act_feats[:, -ctxt_window:],
                prop_feats[:, -ctxt_window:] if prop_feats is not None else None,
            )
            raw_model_next_vid = pred_vid[:, -1:]
            next_prop_feat = pred_prop[:, -1:]

            # Apply constant mean empirical momentum
            grounded_next_vid = vid_feats[:, -1:] + self.static_momentum_delta
            fused_next_vid = (1.0 - rag_lambda) * raw_model_next_vid + rag_lambda * grounded_next_vid

            if getattr(self.model, "normalize_reps", False):
                fused_next_vid = F.normalize(fused_next_vid, p=2, dim=-1)

            vid_feats = torch.cat([vid_feats, fused_next_vid], dim=1)
            prop_feats = torch.cat([prop_feats, next_prop_feat], dim=1)

        return rearrange(vid_feats, "b t v h w d -> b t (v h w) d")

    @torch.no_grad()
    def rollout_rag(
        self,
        z_ctxt: TensorDict,
        act_suffix: torch.Tensor,
        top_k: int = 3,
        rag_lambda: Optional[float] = None,
        temperature: Optional[float] = None,
        adaptive_gating: bool = False,
        similarity_threshold: float = 0.5,
        horizon_decay: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Dense RAG-Augmented Multi-Step Rollout.
        Args:
            z_ctxt: Context TensorDict (visual: [B, ctxt, 1, 16, 16, 384], proprio: [B, ctxt, P_tok, 16])
            act_suffix: Actions [T, B, 20] or [T, B, 4]
            top_k: Retrieval neighbor count (K=0 reduces to baseline)
            rag_lambda: Blend ratio lambda in [0, 1] (lambda=0 is pure neural, lambda=1 is pure retrieval)
            temperature: Softmax weighting temperature
            adaptive_gating: If True, dynamically scales lambda by retrieval similarity to prevent late-horizon drift
            similarity_threshold: Minimum cosine similarity required for memory weighting
            horizon_decay: Optional geometric decay factor (e.g. 0.95^t)
        """
        lam_0 = rag_lambda if rag_lambda is not None else self.rag_lambda

        # Boundary checks: K=0 or lambda=0 -> exact baseline
        if top_k == 0 or lam_0 == 0.0 or self.memory_bank is None or self.memory_bank.num_transitions == 0:
            patches = self.rollout_baseline(z_ctxt, act_suffix)
            return {
                "pred_patches": patches,
                "retrieval_history": [],
                "similarity_scores": [0.0] * act_suffix.shape[0],
                "effective_lambdas": [0.0] * act_suffix.shape[0],
            }

        T, B, A = act_suffix.shape
        ctxt_window = z_ctxt["visual"].shape[1]

        vid_feats = z_ctxt["visual"].clone()
        prop_feats = z_ctxt["proprio"].clone()

        act_suffix_rearranged = rearrange(act_suffix, "t b ... -> b t ...")
        act_feats_suffix = self.model.model.encode_act(act_suffix_rearranged)

        retrieval_history = []
        similarity_scores = []
        effective_lambdas = []

        for h in range(T):
            new_act_feat = act_feats_suffix[:, h : h + 1]
            act_feats = new_act_feat if h == 0 else torch.cat([act_feats, new_act_feat], dim=1)

            # 1. Neural model step prediction
            pred_vid, _, pred_prop = self.model.model.forward_pred(
                vid_feats[:, -ctxt_window:],
                act_feats[:, -ctxt_window:],
                prop_feats[:, -ctxt_window:] if prop_feats is not None else None,
            )
            raw_model_next_vid = pred_vid[:, -1:]
            next_prop_feat = pred_prop[:, -1:]

            # 2. Query Episodic Memory Bank
            current_vid_state = vid_feats[:, -1, 0]  # (B, 16, 16, 384)
            current_raw_act = act_suffix_rearranged[:, h]

            retrieved = self.memory_bank.retrieve(
                query_visual=current_vid_state,
                query_action=current_raw_act,
                top_k=top_k,
                temperature=temperature,
            )
            top1_sim = retrieved["top1_similarity"].mean().item()
            retrieval_history.append(retrieved)
            similarity_scores.append(top1_sim)

            # 3. Compute effective lambda (Adaptive Gating / Decay)
            eff_lambda = lam_0
            if horizon_decay is not None:
                eff_lambda = eff_lambda * (horizon_decay ** h)

            if adaptive_gating:
                # Gate by confidence above threshold
                confidence = max(0.0, (top1_sim - similarity_threshold) / (1.0 - similarity_threshold))
                eff_lambda = eff_lambda * confidence

            eff_lambda = min(max(eff_lambda, 0.0), 1.0)
            effective_lambdas.append(eff_lambda)

            # 4. Dense RAG Fusion
            retrieved_delta = retrieved["weighted_delta"]
            grounded_next_vid = vid_feats[:, -1:] + retrieved_delta
            fused_next_vid = (1.0 - eff_lambda) * raw_model_next_vid + eff_lambda * grounded_next_vid

            if getattr(self.model, "normalize_reps", False):
                fused_next_vid = F.normalize(fused_next_vid, p=2, dim=-1)

            vid_feats = torch.cat([vid_feats, fused_next_vid], dim=1)
            prop_feats = torch.cat([prop_feats, next_prop_feat], dim=1)

        pred_patches = rearrange(vid_feats, "b t v h w d -> b t (v h w) d")

        return {
            "pred_patches": pred_patches,
            "retrieval_history": retrieval_history,
            "similarity_scores": similarity_scores,
            "effective_lambdas": effective_lambdas,
        }
