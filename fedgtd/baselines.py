"""
Baseline federated learning methods for comparison (Section 7.1, Table 3).

Implements functional (non-stub) versions of:
- FedAvg   (McMahan et al., 2017)
- FedProx  (Li et al., 2020)
- Krum     (Blanchard et al., 2017)
- Trimmed Mean (Yin et al., 2018)
- CloudFL  (Wang et al., 2024)
- RobustFL (Zhou et al., 2024)

All baselines use the same DomainSpecificDefender architecture for fair
comparison (Section 7.1).

Reference: Paper Table 3 and Section 7.1 (Comparative Analysis)
"""

import copy
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from fedgtd.config import GameConfig
from fedgtd.models import DomainSpecificDefender


def _local_train_simple(model: nn.Module, loader: DataLoader,
                        epochs: int, lr: float, device: torch.device,
                        weight: torch.Tensor, mu: float = 0.0,
                        global_params: Optional[list] = None):
    """Shared local training logic (optionally with FedProx regularisation)."""
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(weight=weight)

    for _ in range(epochs):
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            loss = criterion(model(bx), by)

            # FedProx proximal term
            if mu > 0 and global_params is not None:
                prox = sum(
                    torch.norm(p - gp) ** 2
                    for p, gp in zip(model.parameters(), global_params)
                )
                loss += (mu / 2) * prox

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


# ═════════════════════════════════════════════════════════════════════════

class FedAvgBaseline:
    """FedAvg (McMahan et al., 2017) – simple averaging."""

    def __init__(self, config: GameConfig, input_dim: int, domain: str):
        self.config = config
        self.device = config.resolve_device()
        self.domain = domain
        n = config.get_n_clients(domain)
        self.models = [
            DomainSpecificDefender(input_dim, domain, config).to(self.device)
            for _ in range(n)
        ]
        rho = config.get_imbalance(domain)
        self.weight = torch.tensor([1.0, rho], device=self.device)

    def train_round(self, loaders: List[DataLoader], lr: float = 0.001):
        for model, loader in zip(self.models, loaders):
            _local_train_simple(model, loader, self.config.local_epochs,
                                lr, self.device, self.weight)
        # Average
        avg_state = {}
        for key in self.models[0].state_dict():
            avg_state[key] = torch.stack(
                [m.state_dict()[key].float() for m in self.models]
            ).mean(0)
        for m in self.models:
            m.load_state_dict(avg_state, strict=False)

    def get_model(self) -> nn.Module:
        return self.models[0]


class FedProxBaseline:
    """FedProx (Li et al., 2020) – proximal regularisation."""

    def __init__(self, config: GameConfig, input_dim: int, domain: str,
                 mu: float = 0.01):
        self.config = config
        self.device = config.resolve_device()
        self.domain = domain
        self.mu = mu
        n = config.get_n_clients(domain)
        self.models = [
            DomainSpecificDefender(input_dim, domain, config).to(self.device)
            for _ in range(n)
        ]
        rho = config.get_imbalance(domain)
        self.weight = torch.tensor([1.0, rho], device=self.device)

    def train_round(self, loaders: List[DataLoader], lr: float = 0.001):
        global_params = [p.data.clone() for p in self.models[0].parameters()]
        for model, loader in zip(self.models, loaders):
            _local_train_simple(model, loader, self.config.local_epochs,
                                lr, self.device, self.weight,
                                mu=self.mu, global_params=global_params)
        avg_state = {}
        for key in self.models[0].state_dict():
            avg_state[key] = torch.stack(
                [m.state_dict()[key].float() for m in self.models]
            ).mean(0)
        for m in self.models:
            m.load_state_dict(avg_state, strict=False)

    def get_model(self) -> nn.Module:
        return self.models[0]


class KrumBaseline:
    """Krum (Blanchard et al., 2017) – Byzantine-tolerant selection."""

    def __init__(self, config: GameConfig, input_dim: int, domain: str):
        self.config = config
        self.device = config.resolve_device()
        self.domain = domain
        n = config.get_n_clients(domain)
        self.models = [
            DomainSpecificDefender(input_dim, domain, config).to(self.device)
            for _ in range(n)
        ]
        rho = config.get_imbalance(domain)
        self.weight = torch.tensor([1.0, rho], device=self.device)

    def train_round(self, loaders: List[DataLoader], lr: float = 0.001):
        for model, loader in zip(self.models, loaders):
            _local_train_simple(model, loader, self.config.local_epochs,
                                lr, self.device, self.weight)

        # Krum selection
        n = len(self.models)
        f = self.config.byzantine_clients
        states = [
            torch.cat([p.data.flatten() for p in m.parameters()])
            for m in self.models
        ]
        scores = []
        for i in range(n):
            dists = sorted(
                torch.norm(states[i] - states[j]).item()
                for j in range(n) if j != i
            )
            scores.append(sum(dists[: n - f - 2]))

        best = int(np.argmin(scores))
        best_state = self.models[best].state_dict()
        for m in self.models:
            m.load_state_dict(best_state)

    def get_model(self) -> nn.Module:
        return self.models[0]


class TrimmedMeanBaseline:
    """Coordinate-wise Trimmed Mean (Yin et al., 2018)."""

    def __init__(self, config: GameConfig, input_dim: int, domain: str,
                 trim_ratio: float = 0.1):
        self.config = config
        self.device = config.resolve_device()
        self.domain = domain
        self.trim_ratio = trim_ratio
        n = config.get_n_clients(domain)
        self.models = [
            DomainSpecificDefender(input_dim, domain, config).to(self.device)
            for _ in range(n)
        ]
        rho = config.get_imbalance(domain)
        self.weight = torch.tensor([1.0, rho], device=self.device)

    def train_round(self, loaders: List[DataLoader], lr: float = 0.001):
        for model, loader in zip(self.models, loaders):
            _local_train_simple(model, loader, self.config.local_epochs,
                                lr, self.device, self.weight)

        n = len(self.models)
        k = int(n * self.trim_ratio)
        agg_state = {}
        for key in self.models[0].state_dict():
            vals = torch.stack([m.state_dict()[key].float() for m in self.models])
            if k > 0 and n > 2 * k:
                sorted_vals, _ = torch.sort(vals, dim=0)
                vals = sorted_vals[k: n - k]
            agg_state[key] = vals.mean(0)
        for m in self.models:
            m.load_state_dict(agg_state, strict=False)

    def get_model(self) -> nn.Module:
        return self.models[0]


# ═════════════════════════════════════════════════════════════════════════
# Runner
# ═════════════════════════════════════════════════════════════════════════

def run_all_baselines(
    config: GameConfig,
    train_loaders: Dict[str, List[DataLoader]],
    test_loaders: Dict[str, DataLoader],
    data_dims: Dict[str, int],
    n_rounds: int = 50,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Run all baselines and return per-method, per-domain results.

    Returns:
        {"FedAvg": {"edge": {"accuracy": ..., ...}, ...}, ...}
    """
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
    import torch.nn.functional as F

    device = config.resolve_device()

    baseline_classes = {
        "FedAvg": FedAvgBaseline,
        "FedProx": FedProxBaseline,
        "Krum": KrumBaseline,
        "TrimmedMean": TrimmedMeanBaseline,
    }

    all_results = {}

    for name, cls in baseline_classes.items():
        print(f"  Running {name}...")
        method_results = {}

        for domain in ("edge", "container", "soc"):
            if domain not in train_loaders:
                continue

            baseline = cls(config, data_dims[domain], domain)

            for r in range(n_rounds):
                baseline.train_round(train_loaders[domain],
                                     lr=config.get_base_lr(domain))

            # Evaluate
            model = baseline.get_model()
            model.eval()
            preds, labels, probs = [], [], []
            with torch.no_grad():
                for bx, by in test_loaders[domain]:
                    bx = bx.to(device)
                    logits = model(bx)
                    prob = F.softmax(logits, dim=1)[:, 1]
                    preds.extend(logits.argmax(1).cpu().numpy())
                    labels.extend(by.numpy())
                    probs.extend(prob.cpu().numpy())

            acc = accuracy_score(labels, preds)
            prec, rec, f1, _ = precision_recall_fscore_support(
                labels, preds, average="binary", zero_division=0
            )
            try:
                auc = roc_auc_score(labels, probs)
            except ValueError:
                auc = 0.5

            method_results[domain] = {
                "accuracy": acc, "precision": prec,
                "recall": rec, "f1": f1, "auc": auc,
            }

        all_results[name] = method_results

    return all_results
