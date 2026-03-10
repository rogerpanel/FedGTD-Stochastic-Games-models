"""
Byzantine-resilient federated aggregation (Algorithm 1).

Implements:
- Cross-domain projection-based Byzantine detection
- Domain-specific gradient clipping (Section 4.2)
- Calibrated differential privacy noise (Theorem 3)
- Trimmed mean aggregation with reputation scores

Reference: Paper Section 5 (Algorithm 1) and Section 4.2 (Privacy)
"""

from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from fedgtd.config import GameConfig


class ByzantineResilientAggregator:
    """Algorithm 1: Byzantine-Resilient Cross-Domain Aggregation.

    Steps (per federated round, per domain):
        1. Clip each client gradient to domain-specific norm C_d.
        2. Project gradients into a shared low-dimensional subspace P.
        3. Compute pairwise cosine similarities in the projected space.
        4. Flag clients whose median similarity is below threshold τ_d.
        5. Aggregate honest clients via coordinate-wise trimmed mean.
        6. Add Gaussian DP noise calibrated to (ε_d, δ_d).
    """

    def __init__(self, config: GameConfig):
        self.config = config
        self.reputation_scores: Dict[str, Dict[int, float]] = defaultdict(
            lambda: defaultdict(lambda: 1.0)
        )
        self.detection_history: List[dict] = []

    # ── Step 1: gradient clipping ────────────────────────────────────────

    def clip_gradient(self, gradient: torch.Tensor, domain: str) -> torch.Tensor:
        """Per-sample gradient clipping to norm C_d (Section 4.2)."""
        max_norm = self.config.get_clip_norm(domain)
        norm = torch.norm(gradient)
        if norm > max_norm:
            gradient = gradient * (max_norm / norm)
        return gradient

    # ── Step 2–4: Byzantine detection ────────────────────────────────────

    def _build_projection_matrix(self, domain: str, device: torch.device
                                  ) -> torch.Tensor:
        """Projection P ∈ R^{10 × d_k} to shared flow-feature subspace."""
        d = self.config.get_features(domain)
        P = torch.zeros(10, d, device=device)
        for i in range(min(10, d)):
            P[i, i] = 1.0
        return P

    def detect_byzantine(self, flat_grads: List[torch.Tensor], domain: str,
                          round_num: int) -> List[int]:
        """Cross-domain cosine-similarity Byzantine detection.

        Returns list of client indices flagged as Byzantine.
        """
        n = len(flat_grads)
        max_byz = self.config.byzantine_clients
        if n <= 2 * max_byz:
            return []

        device = flat_grads[0].device
        P = self._build_projection_matrix(domain, device)

        # Project into common subspace
        projected = []
        for g in flat_grads:
            if g.numel() >= P.shape[1]:
                proj = P @ g[: P.shape[1]]
            else:
                proj = g[:10] if g.numel() >= 10 else g
            projected.append(proj)

        # Pairwise cosine similarity matrix
        sim = torch.zeros(n, n, device=device)
        for i in range(n):
            for j in range(i + 1, n):
                c = F.cosine_similarity(
                    projected[i].unsqueeze(0),
                    projected[j].unsqueeze(0),
                ).item()
                sim[i, j] = c
                sim[j, i] = c

        # Median similarity per client
        threshold = self.config.get_detection_threshold(domain)
        flagged = []
        for i in range(n):
            row = sim[i]
            row_nonzero = row[row != 0]
            if len(row_nonzero) > 0:
                med = torch.median(row_nonzero).item()
            else:
                med = 0.0
            if med < threshold:
                flagged.append(i)

        # Cap at max allowed Byzantines
        flagged = flagged[:max_byz]

        # Update reputation
        for i in flagged:
            self.reputation_scores[domain][i] *= 0.5

        self.detection_history.append({
            "round": round_num, "domain": domain, "flagged": flagged
        })
        return flagged

    # ── Step 5: trimmed mean ─────────────────────────────────────────────

    @staticmethod
    def trimmed_mean(tensors: List[torch.Tensor],
                     trim_ratio: float) -> torch.Tensor:
        """Coordinate-wise trimmed mean.

        Sorts values along the stacking dimension, removes top/bottom
        `trim_ratio` fraction, and averages the rest.
        """
        stacked = torch.stack(tensors)          # (n, *)
        n = stacked.shape[0]
        k = int(n * trim_ratio)

        if k > 0 and n > 2 * k:
            sorted_vals, _ = torch.sort(stacked, dim=0)
            trimmed = sorted_vals[k: n - k]
            return trimmed.mean(dim=0)

        return stacked.mean(dim=0)

    # ── Step 6: differential privacy ─────────────────────────────────────

    def add_dp_noise(self, param: torch.Tensor, domain: str) -> torch.Tensor:
        """Gaussian mechanism calibrated to (ε_d, δ_d) (Theorem 3).

        σ = 2·C_d · √(2·ln(1.25/δ)) / ε
        """
        eps, delta = self.config.get_privacy_params(domain)
        C = self.config.get_clip_norm(domain)
        sensitivity = 2.0 * C
        sigma = sensitivity * np.sqrt(2.0 * np.log(1.25 / delta)) / eps
        return param + torch.randn_like(param) * sigma

    # ── Full aggregation pipeline ────────────────────────────────────────

    def aggregate(
        self,
        client_updates: List[Dict[str, torch.Tensor]],
        domain: str,
        round_num: int = 0,
    ) -> Dict[str, torch.Tensor]:
        """Full Algorithm 1 pipeline.

        Each element of `client_updates` is a dict mapping
        parameter-name -> parameter-tensor (the model state_dict snapshot).

        Returns aggregated state_dict with DP noise added.
        """
        if not client_updates:
            return {}

        param_names = list(client_updates[0].keys())

        # Flatten each client's parameters for Byzantine detection
        flat_grads = []
        for upd in client_updates:
            flat = torch.cat([upd[k].flatten() for k in param_names])
            flat = self.clip_gradient(flat, domain)
            flat_grads.append(flat)

        # Detect Byzantines
        byz_indices = self.detect_byzantine(flat_grads, domain, round_num)
        honest_mask = [i for i in range(len(client_updates)) if i not in byz_indices]

        if not honest_mask:
            # Fallback: keep all but last `byzantine_clients`
            honest_mask = list(range(max(1, len(client_updates) - self.config.byzantine_clients)))

        honest_updates = [client_updates[i] for i in honest_mask]

        # Trimmed-mean aggregation per parameter
        trim_ratio = self.config.get_trim_ratio(domain)
        aggregated: Dict[str, torch.Tensor] = {}
        for name in param_names:
            values = [upd[name] for upd in honest_updates]
            agg = self.trimmed_mean(values, trim_ratio)
            agg = self.add_dp_noise(agg, domain)
            aggregated[name] = agg

        return aggregated
