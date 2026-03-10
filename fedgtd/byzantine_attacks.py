"""
Byzantine attack implementations for resilience testing (Section 6.3, Reviewer 2).

Implements the four attack types tested in the paper:
1. Label flipping      – swaps labels 0 <-> 1
2. Gradient scaling    – scales gradients by 3–5x
3. Backdoor injection  – embeds trigger pattern
4. Model poisoning     – replaces model with random parameters

Plus the three advanced attacks requested by Reviewer 2:
5. Adaptive attack     – mimics honest gradient statistics (93.1% retention)
6. Colluding attack    – coordinated among f clients (92.4% retention)
7. Stealthy attack     – slowly drifts model over rounds (91.7% retention)

Reference: Paper Section 5.2 (Threat Model) and Section 6.3 (Advanced Attacks)
"""

from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn


class ByzantineAttacker:
    """Generate Byzantine client updates for resilience testing."""

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)

    # ── Basic attacks (Section 5.2) ──────────────────────────────────────

    def label_flip(self, state_dict: Dict[str, torch.Tensor],
                   honest_states: List[Dict[str, torch.Tensor]]
                   ) -> Dict[str, torch.Tensor]:
        """Label-flipping attack: negate the gradient direction.

        Simulates the effect of training on flipped labels by returning
        the negative of the honest mean plus twice the attacker's state.
        """
        if not honest_states:
            return state_dict

        mean_state = {}
        for key in state_dict:
            vals = torch.stack([s[key] for s in honest_states])
            mean_state[key] = vals.mean(dim=0)

        poisoned = {}
        for key in state_dict:
            poisoned[key] = 2 * state_dict[key] - mean_state[key]
        return poisoned

    def gradient_scaling(self, state_dict: Dict[str, torch.Tensor],
                         scale: float = 5.0) -> Dict[str, torch.Tensor]:
        """Gradient-scaling attack: amplify parameters by factor 3–5x.

        Pushes the aggregated model away from the honest optimum.
        """
        return {k: v * scale for k, v in state_dict.items()}

    def backdoor_injection(self, state_dict: Dict[str, torch.Tensor],
                           trigger_magnitude: float = 2.0
                           ) -> Dict[str, torch.Tensor]:
        """Backdoor trigger injection.

        Embeds a trigger pattern in the first-layer weights.
        """
        poisoned = {k: v.clone() for k, v in state_dict.items()}
        for key in poisoned:
            if "fc1" in key or "backbone.0" in key:
                trigger = torch.randn_like(poisoned[key]) * trigger_magnitude
                # Only modify a small region (10% of weights)
                mask = torch.rand_like(poisoned[key]) < 0.1
                poisoned[key] = poisoned[key] + trigger * mask.float()
                break
        return poisoned

    def model_poisoning(self, state_dict: Dict[str, torch.Tensor]
                        ) -> Dict[str, torch.Tensor]:
        """Full model poisoning: replace with random parameters."""
        return {k: torch.randn_like(v) for k, v in state_dict.items()}

    # ── Advanced attacks (Reviewer 2, Section 6.3) ───────────────────────

    def adaptive_attack(self, state_dict: Dict[str, torch.Tensor],
                        honest_states: List[Dict[str, torch.Tensor]],
                        stealth_factor: float = 0.8
                        ) -> Dict[str, torch.Tensor]:
        """Adaptive attack: mimics honest gradient statistics.

        The attacker computes the honest mean and standard deviation,
        then injects a perturbation within the expected range but in
        a harmful direction.

        Paper result: 93.1% performance retention (detected effectively).
        """
        if not honest_states:
            return self.model_poisoning(state_dict)

        poisoned = {}
        for key in state_dict:
            honest_vals = torch.stack([s[key] for s in honest_states])
            mean = honest_vals.mean(dim=0)
            std = honest_vals.std(dim=0) + 1e-8

            # Stay within stealth_factor * std of the mean
            # but push in the opposite direction
            direction = mean - state_dict[key]
            direction = direction / (torch.norm(direction) + 1e-8)
            perturbation = direction * std * stealth_factor

            poisoned[key] = mean - perturbation

        return poisoned

    def colluding_attack(self, state_dicts: List[Dict[str, torch.Tensor]],
                         honest_states: List[Dict[str, torch.Tensor]],
                         target_direction: Optional[torch.Tensor] = None
                         ) -> List[Dict[str, torch.Tensor]]:
        """Colluding attack: coordinated among f Byzantine clients.

        All colluding clients submit updates that pull the aggregated
        model in the same adversarial direction, but each with slight
        noise to avoid detection.

        Paper result: 92.4% performance retention.
        """
        if not honest_states:
            return [self.model_poisoning(sd) for sd in state_dicts]

        n_colluders = len(state_dicts)
        poisoned_list = []

        for key in state_dicts[0]:
            honest_vals = torch.stack([s[key] for s in honest_states])
            mean = honest_vals.mean(dim=0)

            if target_direction is None:
                direction = -mean / (torch.norm(mean) + 1e-8)
            else:
                direction = target_direction

        for i, sd in enumerate(state_dicts):
            poisoned = {}
            for key in sd:
                honest_vals = torch.stack([s[key] for s in honest_states])
                mean = honest_vals.mean(dim=0)
                std = honest_vals.std(dim=0) + 1e-8

                # Coordinated direction with individual noise
                noise = torch.randn_like(mean) * std * 0.1
                perturbation = -mean * 0.3 + noise * (i + 1) / n_colluders
                poisoned[key] = mean + perturbation

            poisoned_list.append(poisoned)

        return poisoned_list

    def stealthy_attack(self, state_dict: Dict[str, torch.Tensor],
                        honest_states: List[Dict[str, torch.Tensor]],
                        round_num: int, total_rounds: int = 200,
                        max_drift: float = 0.5
                        ) -> Dict[str, torch.Tensor]:
        """Stealthy slow-drift attack: gradually increases poisoning.

        In early rounds, the attacker behaves honestly. Over time,
        it slowly increases the magnitude of its adversarial perturbation.

        Paper result: 91.7% performance retention.
        """
        if not honest_states:
            return state_dict

        # Drift factor increases linearly over rounds
        drift_factor = min(max_drift, max_drift * round_num / total_rounds)

        poisoned = {}
        for key in state_dict:
            honest_vals = torch.stack([s[key] for s in honest_states])
            mean = honest_vals.mean(dim=0)
            std = honest_vals.std(dim=0) + 1e-8

            # Small, slowly increasing perturbation
            perturbation = -std * drift_factor
            poisoned[key] = mean + perturbation

        return poisoned
