"""
Adversarial robustness evaluation (Section 7.3).

Implements:
- FGSM (Goodfellow et al., 2015)
- PGD  (Madry et al., 2018)
- C&W  (Carlini & Wagner, 2017) – simplified L2 variant

Reference: Paper Section 7.3 (Adversarial Robustness Analysis)
"""

from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


class AdversarialEvaluator:
    """Evaluate model robustness under gradient-based attacks."""

    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model
        self.device = device

    # ── FGSM ─────────────────────────────────────────────────────────────

    def fgsm(self, x: torch.Tensor, y: torch.Tensor,
             epsilon: float = 0.1) -> torch.Tensor:
        """Fast Gradient Sign Method.

        δ = ε · sign(∇_x L(θ, x, y))
        """
        x_adv = x.clone().detach().requires_grad_(True)
        loss = F.cross_entropy(self.model(x_adv), y)
        loss.backward()
        return (x + epsilon * x_adv.grad.sign()).detach()

    # ── PGD ──────────────────────────────────────────────────────────────

    def pgd(self, x: torch.Tensor, y: torch.Tensor,
            epsilon: float = 0.1, steps: int = 10,
            alpha: float = 0.01) -> torch.Tensor:
        """Projected Gradient Descent (L∞ ball).

        x_{t+1} = Proj_{B_∞(x, ε)} [ x_t + α · sign(∇_x L) ]
        """
        x_adv = x.clone().detach()
        for _ in range(steps):
            x_adv.requires_grad_(True)
            loss = F.cross_entropy(self.model(x_adv), y)
            loss.backward()
            grad_sign = x_adv.grad.sign()
            x_adv = (x_adv + alpha * grad_sign).detach()
            # Project back into ε-ball
            x_adv = torch.max(torch.min(x_adv, x + epsilon), x - epsilon)
        return x_adv

    # ── C&W (simplified L2) ──────────────────────────────────────────────

    def cw_l2(self, x: torch.Tensor, y: torch.Tensor,
              c: float = 1.0, steps: int = 50,
              lr: float = 0.01) -> torch.Tensor:
        """Simplified Carlini–Wagner L2 attack.

        min ‖δ‖² + c · max(Z(x+δ)_y − max_{j≠y} Z(x+δ)_j, 0)
        """
        delta = torch.zeros_like(x, requires_grad=True)
        optimizer = torch.optim.Adam([delta], lr=lr)

        for _ in range(steps):
            x_adv = x + delta
            logits = self.model(x_adv)

            # Objective: push logit of true class below runner-up
            one_hot = F.one_hot(y, logits.shape[1]).float()
            real = (logits * one_hot).sum(dim=1)
            other = ((1 - one_hot) * logits - one_hot * 1e4).max(dim=1).values
            f_loss = torch.clamp(real - other, min=0).mean()

            l2_loss = torch.norm(delta.view(delta.shape[0], -1), dim=1).mean()
            loss = l2_loss + c * f_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return (x + delta).detach()

    # ── Full evaluation ──────────────────────────────────────────────────

    def evaluate(self, loader: DataLoader,
                 epsilons: List[float] = (0.01, 0.05, 0.1, 0.2),
                 max_samples: int = 2000,
                 ) -> Dict[str, Dict[str, float]]:
        """Evaluate clean + adversarial accuracy at each epsilon.

        Returns:
            {
                "eps_0.01": {"clean": 0.95, "fgsm": 0.88, "pgd": 0.82, "cw": 0.85},
                "eps_0.05": { ... },
                ...
            }
        """
        self.model.eval()
        results = {}

        for eps in epsilons:
            clean_correct = fgsm_correct = pgd_correct = cw_correct = 0
            total = 0

            for bx, by in loader:
                bx, by = bx.to(self.device), by.to(self.device)

                # Clean
                with torch.no_grad():
                    clean_correct += (self.model(bx).argmax(1) == by).sum().item()

                # FGSM
                x_fgsm = self.fgsm(bx, by, eps)
                with torch.no_grad():
                    fgsm_correct += (self.model(x_fgsm).argmax(1) == by).sum().item()

                # PGD
                x_pgd = self.pgd(bx, by, eps)
                with torch.no_grad():
                    pgd_correct += (self.model(x_pgd).argmax(1) == by).sum().item()

                # C&W
                x_cw = self.cw_l2(bx, by, c=1.0, steps=20)
                with torch.no_grad():
                    cw_correct += (self.model(x_cw).argmax(1) == by).sum().item()

                total += by.size(0)
                if total >= max_samples:
                    break

            results[f"eps_{eps}"] = {
                "clean": clean_correct / max(total, 1),
                "fgsm": fgsm_correct / max(total, 1),
                "pgd": pgd_correct / max(total, 1),
                "cw": cw_correct / max(total, 1),
            }

        return results
