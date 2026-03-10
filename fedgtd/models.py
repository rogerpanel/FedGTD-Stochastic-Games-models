"""
Neural network architectures for FedGTD.

Implements:
- ResidualBlock with LayerNorm (Section 4.4)
- DomainSpecificDefender: per-domain classifiers with residual blocks
- StrategicAdversaryNetwork: attention-based adversarial perturbation generator

Reference: Paper Section 4.4 (Neural Architecture) and Section 3.2 (Adversary Model)
"""

import torch
import torch.nn as nn
from fedgtd.config import GameConfig


class ResidualBlock(nn.Module):
    """Pre-activation residual block with LayerNorm.

    Architecture per paper Section 4.4:
        x -> Linear -> LayerNorm -> LeakyReLU -> Dropout -> Linear -> LayerNorm -> (+skip) -> LeakyReLU
    """

    def __init__(self, in_features: int, out_features: int,
                 dropout: float = 0.3, slope: float = 0.01):
        super().__init__()
        self.fc1 = nn.Linear(in_features, out_features)
        self.ln1 = nn.LayerNorm(out_features)
        self.act = nn.LeakyReLU(slope)
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(out_features, out_features)
        self.ln2 = nn.LayerNorm(out_features)

        self.skip = (nn.Linear(in_features, out_features)
                     if in_features != out_features else nn.Identity())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        out = self.drop(self.act(self.ln1(self.fc1(x))))
        out = self.ln2(self.fc2(out))
        return self.act(out + residual)


class DomainSpecificDefender(nn.Module):
    """Domain-specific defender network (Section 4.4).

    Each domain has its own depth and width:
        Edge/Container : [512, 256, 128, 64, 32]
        SOC            : [256, 128, 64, 32]

    The network has two heads:
        - multiclass_head: for fine-grained attack-family classification
        - binary_head:     for benign-vs-attack binary detection (primary task)
    """

    def __init__(self, input_dim: int, domain: str, config: GameConfig):
        super().__init__()
        self.domain = domain
        hidden_dims = config.get_hidden_dims(domain)
        n_attack_classes = config.get_n_attack_classes(domain)

        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(ResidualBlock(prev, h, config.dropout, config.leaky_relu_slope))
            prev = h
        self.backbone = nn.Sequential(*layers)

        self.multiclass_head = nn.Linear(prev, n_attack_classes)
        self.binary_head = nn.Linear(prev, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Binary classification logits (default forward)."""
        return self.binary_head(self.backbone(x))

    def forward_multiclass(self, x: torch.Tensor) -> torch.Tensor:
        """Fine-grained attack-family logits."""
        return self.multiclass_head(self.backbone(x))

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract penultimate features for game-theoretic analysis."""
        return self.backbone(x)


class StrategicAdversaryNetwork(nn.Module):
    """Strategic adversary with attention-based perturbation (Section 3.2).

    Components:
        1. Feature-importance attention: learns which features to perturb.
        2. Perturbation generator: bounded Tanh output scaled by ε.
        3. Strategy network: outputs mixed-strategy distribution over |A_a| actions.
    """

    def __init__(self, input_dim: int, domain: str, config: GameConfig):
        super().__init__()
        self.domain = domain
        h = config.adversary_hidden_dim
        self.epsilon = config.get_adversary_epsilon(domain)
        n_strategies = config.n_strategies

        # Attention for feature importance
        self.attention = nn.Sequential(
            nn.Linear(input_dim, h),
            nn.Tanh(),
            nn.Linear(h, input_dim),
            nn.Softmax(dim=-1),
        )

        # Perturbation generator (output in [-ε, ε])
        self.generator = nn.Sequential(
            nn.Linear(input_dim, h),
            nn.ReLU(),
            nn.LayerNorm(h),
            nn.Linear(h, h),
            nn.ReLU(),
            nn.LayerNorm(h),
            nn.Linear(h, input_dim),
            nn.Tanh(),
        )

        # Mixed-strategy selector
        self.strategy_net = nn.Sequential(
            nn.Linear(input_dim, h),
            nn.ReLU(),
            nn.Linear(h, n_strategies),
            nn.Softmax(dim=-1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Generate adversarial perturbations δ with ‖δ‖_∞ ≤ ε."""
        att = self.attention(x)
        pert = self.generator(x)
        return pert * att * self.epsilon

    def get_strategy(self, x: torch.Tensor) -> torch.Tensor:
        """Return adversarial mixed-strategy distribution π_a ∈ Δ^{|A_a|}."""
        return self.strategy_net(x)
