"""
FedGTD: Byzantine-Resilient Stochastic Games for Federated Multi-Cloud Intrusion Detection

Reference: Anaedevha et al., "Byzantine-Resilient Stochastic Games for Federated
Multi-Cloud Intrusion Detection", Complex and Intelligent Systems (2026).

This package implements the complete FedGTD framework including:
- Domain-specific defender networks (Edge-IIoT, Container, SOC)
- Stochastic differential game dynamics with Nash equilibrium computation
- Byzantine-resilient federated aggregation with cross-domain detection
- Martingale-based convergence analysis with heterogeneous Lyapunov functions
- Differential privacy with domain-calibrated noise
- Adversarial robustness evaluation (FGSM, PGD, C&W)
"""

__version__ = "2.0.0"

from fedgtd.config import GameConfig
from fedgtd.datasets import ICS3DDataHandler
from fedgtd.models import DomainSpecificDefender, StrategicAdversaryNetwork
from fedgtd.aggregation import ByzantineResilientAggregator
from fedgtd.game_dynamics import StochasticDifferentialGame, NashEquilibriumSolver
from fedgtd.convergence import MartingaleConvergenceAnalyzer
from fedgtd.federated import FedGTDSystem
from fedgtd.adversarial import AdversarialEvaluator
from fedgtd.byzantine_attacks import ByzantineAttacker
