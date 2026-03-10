"""
Configuration and hyperparameters for FedGTD.

All parameters are aligned with the paper:
- Section 3: Federation parameters, domain specifications
- Section 4: Game-theoretic parameters, privacy budgets
- Section 6: Experimental setup, learning rates, clipping norms
- Table 2: Hyperparameter summary
"""

from dataclasses import dataclass, field
from typing import Dict, Tuple
import torch


@dataclass
class GameConfig:
    """Complete configuration aligned with paper Section 6 (Table 2)."""

    # ── Federation topology (Section 3.1, Definition 5) ──────────────────
    n_defenders: int = 20          # K = 20 organizations
    n_edge_clients: int = 7        # Edge-IIoT cloud segment
    n_container_clients: int = 7   # Container cloud segment
    n_soc_clients: int = 6         # SOC cloud segment
    cross_domain_clients: int = 3  # Clients spanning multiple domains

    # ── Domain-specific feature dimensions (Definition 5) ────────────────
    edge_features: int = 63        # Edge-IIoT: averaged from 60–140 protocol features
    container_features: int = 87   # Container: CICFlowMeter + syscall features
    soc_features: int = 46         # SOC: GUIDE entity/alert features

    # ── Attack families per domain ───────────────────────────────────────
    edge_attacks: int = 14         # 14 attack families (DDoS, Ransomware, etc.)
    container_attacks: int = 11    # 11 CVE exploit categories
    soc_entities: int = 33         # 33 entity types for triage

    # ── Class imbalance ratios (Section 3.3, Table 1) ────────────────────
    edge_imbalance: float = 2.67   # ρ_edge: benign-to-attack ratio
    container_imbalance: float = 15.7  # ρ_container
    soc_imbalance: float = 99.0    # ρ_soc: extreme imbalance

    # ── Game-theoretic parameters (Section 4) ────────────────────────────
    discount_factor: float = 0.95  # γ in discounted stochastic game
    nash_threshold: float = 1e-4   # ε for Nash gap convergence criterion
    n_strategies: int = 10         # |A_d| = |A_a| = 10 discrete strategies

    # ── Learning parameters (Section 6.4, Table 2) ──────────────────────
    max_rounds: int = 200          # T_max federated rounds
    local_epochs: int = 5          # E local SGD epochs per round
    batch_size_edge: int = 256
    batch_size_container: int = 256
    batch_size_soc: int = 1024

    # Base learning rates (η_d^(0), adapted by domain imbalance)
    lr_edge: float = 0.001
    lr_container: float = 0.0005
    lr_soc: float = 0.0001
    lr_decay_power: float = 2 / 3  # η_d(t) = η_d^(0) · √ρ_d / (t+1)^{2/3}

    weight_decay: float = 1e-4     # AdamW weight decay

    # ── Differential privacy (Definition 6, Theorem 3) ───────────────────
    epsilon_edge: float = 2.5
    delta_edge: float = 1e-5
    epsilon_container: float = 2.0
    delta_container: float = 1e-6
    epsilon_soc: float = 1.8
    delta_soc: float = 1e-7

    # ── Gradient clipping norms (Section 6.4) ────────────────────────────
    clip_norm_edge: float = 0.61
    clip_norm_container: float = 0.13
    clip_norm_soc: float = 0.01

    # ── Byzantine parameters (Section 5) ─────────────────────────────────
    byzantine_fraction: float = 0.15  # f < 1/3 of clients per domain
    byzantine_clients: int = 3        # max Byzantine per domain
    trim_ratio_edge: float = 0.1
    trim_ratio_container: float = 0.15
    trim_ratio_soc: float = 0.2

    # Cross-domain detection thresholds (Algorithm 1)
    detection_threshold_edge: float = 0.5
    detection_threshold_container: float = 0.6
    detection_threshold_soc: float = 0.7

    # ── Neural architecture (Section 4.4) ────────────────────────────────
    hidden_dims_edge: tuple = (512, 256, 128, 64, 32)
    hidden_dims_container: tuple = (512, 256, 128, 64, 32)
    hidden_dims_soc: tuple = (256, 128, 64, 32)
    dropout: float = 0.3
    leaky_relu_slope: float = 0.01

    # ── Adversary network ────────────────────────────────────────────────
    adversary_hidden_dim: int = 128
    adversary_epsilon_edge: float = 0.1
    adversary_epsilon_container: float = 0.1
    adversary_epsilon_soc: float = 0.05
    adversarial_mix_ratio: float = 0.5  # α for clean/adversarial loss mixing

    # ── SDE parameters (Section 4.1.2, Equations 7–9) ───────────────────
    sde_dt: float = 0.01           # Time step for Euler–Maruyama discretisation
    jump_rate_edge: float = 0.01   # Poisson jump intensity λ_edge
    jump_rate_container: float = 0.008
    jump_rate_soc: float = 0.005
    jump_magnitude: float = 0.1    # Jump size scaling

    # ── Reproducibility ──────────────────────────────────────────────────
    seed: int = 42
    device: str = "auto"           # "auto", "cuda", or "cpu"

    # ── Dirichlet non-IID (Section 6.3) ──────────────────────────────────
    dirichlet_alpha: float = 0.3   # α for Dirichlet-based heterogeneous split

    def resolve_device(self) -> torch.device:
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)

    def get_privacy_params(self, domain: str) -> Tuple[float, float]:
        return {
            "edge": (self.epsilon_edge, self.delta_edge),
            "container": (self.epsilon_container, self.delta_container),
            "soc": (self.epsilon_soc, self.delta_soc),
        }[domain]

    def get_clip_norm(self, domain: str) -> float:
        return {
            "edge": self.clip_norm_edge,
            "container": self.clip_norm_container,
            "soc": self.clip_norm_soc,
        }[domain]

    def get_imbalance(self, domain: str) -> float:
        return {
            "edge": self.edge_imbalance,
            "container": self.container_imbalance,
            "soc": self.soc_imbalance,
        }[domain]

    def get_batch_size(self, domain: str) -> int:
        return {
            "edge": self.batch_size_edge,
            "container": self.batch_size_container,
            "soc": self.batch_size_soc,
        }[domain]

    def get_base_lr(self, domain: str) -> float:
        return {
            "edge": self.lr_edge,
            "container": self.lr_container,
            "soc": self.lr_soc,
        }[domain]

    def get_hidden_dims(self, domain: str) -> tuple:
        return {
            "edge": self.hidden_dims_edge,
            "container": self.hidden_dims_container,
            "soc": self.hidden_dims_soc,
        }[domain]

    def get_n_clients(self, domain: str) -> int:
        return {
            "edge": self.n_edge_clients,
            "container": self.n_container_clients,
            "soc": self.n_soc_clients,
        }[domain]

    def get_adversary_epsilon(self, domain: str) -> float:
        return {
            "edge": self.adversary_epsilon_edge,
            "container": self.adversary_epsilon_container,
            "soc": self.adversary_epsilon_soc,
        }[domain]

    def get_trim_ratio(self, domain: str) -> float:
        return {
            "edge": self.trim_ratio_edge,
            "container": self.trim_ratio_container,
            "soc": self.trim_ratio_soc,
        }[domain]

    def get_detection_threshold(self, domain: str) -> float:
        return {
            "edge": self.detection_threshold_edge,
            "container": self.detection_threshold_container,
            "soc": self.detection_threshold_soc,
        }[domain]

    def get_jump_rate(self, domain: str) -> float:
        return {
            "edge": self.jump_rate_edge,
            "container": self.jump_rate_container,
            "soc": self.jump_rate_soc,
        }[domain]

    def get_features(self, domain: str) -> int:
        return {
            "edge": self.edge_features,
            "container": self.container_features,
            "soc": self.soc_features,
        }[domain]

    def get_n_attack_classes(self, domain: str) -> int:
        return {
            "edge": self.edge_attacks,
            "container": self.container_attacks,
            "soc": self.soc_entities,
        }[domain]
