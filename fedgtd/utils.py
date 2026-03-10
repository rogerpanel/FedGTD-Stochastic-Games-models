"""
Utility functions for reproducibility, seeding, and logging.
"""

import numpy as np
import torch
import random
import time
from typing import Dict, Any


def set_seeds(seed: int = 42):
    """Set all random seeds for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class MetricsTracker:
    """Tracks and stores experiment metrics across rounds."""

    def __init__(self):
        self.round_metrics = []
        self.domain_metrics = {"edge": [], "container": [], "soc": []}
        self.convergence_metrics = []
        self.byzantine_detection_log = []
        self.privacy_budget_spent = {"edge": 0.0, "container": 0.0, "soc": 0.0}

    def log_round(self, metrics: Dict[str, Any]):
        self.round_metrics.append(metrics)

    def log_domain(self, domain: str, metrics: Dict[str, Any]):
        self.domain_metrics[domain].append(metrics)

    def log_convergence(self, metrics: Dict[str, Any]):
        self.convergence_metrics.append(metrics)

    def log_byzantine_detection(self, domain: str, detected_indices: list,
                                 round_num: int):
        self.byzantine_detection_log.append({
            "round": round_num,
            "domain": domain,
            "detected": detected_indices,
        })

    def get_last_round(self) -> Dict[str, Any]:
        return self.round_metrics[-1] if self.round_metrics else {}

    def summary(self) -> Dict[str, Any]:
        if not self.round_metrics:
            return {}
        last = self.round_metrics[-1]
        return {
            "total_rounds": len(self.round_metrics),
            "final_accuracy": last.get("avg_accuracy"),
            "final_loss": last.get("avg_loss"),
            "final_nash_gap": last.get("nash_gap"),
            "converged": last.get("converged", False),
        }


class Timer:
    """Simple context manager timer."""

    def __init__(self, label: str = ""):
        self.label = label
        self.elapsed = 0.0

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.elapsed = time.time() - self.start
        if self.label:
            print(f"  [{self.label}] {self.elapsed:.2f}s")
