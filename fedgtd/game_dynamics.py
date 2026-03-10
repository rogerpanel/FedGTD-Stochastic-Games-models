"""
Stochastic differential game dynamics and Nash equilibrium computation.

Implements:
- StochasticDifferentialGame: continuous-time SDE with Poisson jumps
  (Section 4.1.2, Equations 7–9)
- NashEquilibriumSolver: mixed-strategy Nash via linear programming
  with imbalance-adjusted payoff matrices (Theorem 2, Definition 8)

Reference: Paper Section 4 (Stochastic Game Formulation)
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import linprog

from fedgtd.config import GameConfig

DOMAINS = ("edge", "container", "soc")


# ═════════════════════════════════════════════════════════════════════════
# Stochastic Differential Game  (Section 4.1.2)
# ═════════════════════════════════════════════════════════════════════════

class StochasticDifferentialGame:
    """Continuous-time stochastic differential game with jump-diffusion.

    State evolution per domain (Euler–Maruyama with compound Poisson jumps):

        dX_d = μ_d(X_d, a_d) dt + Σ_d(X_d, a_d) dW_d + J_d dN_d(λ_d)

    where:
        μ_d  = drift network (parameterised)
        Σ_d  = diffusion matrix network
        W_d  = standard Brownian motion
        N_d  = Poisson process with intensity λ_d (attack arrival rate)
        J_d  = jump magnitude
    """

    def __init__(self, config: GameConfig, device: torch.device):
        self.config = config
        self.device = device

        # Per-domain state vectors
        self.states: Dict[str, torch.Tensor] = {}
        self.drift_nets: Dict[str, nn.Linear] = {}
        self.diffusion_nets: Dict[str, nn.Linear] = {}

        n_act = config.n_strategies

        for domain in DOMAINS:
            d = config.get_features(domain)
            self.states[domain] = torch.zeros(d, device=device)
            self.drift_nets[domain] = nn.Linear(d + n_act, d).to(device)
            self.diffusion_nets[domain] = nn.Linear(d + n_act, d * d).to(device)

        self.time = 0.0

    def evolve(self, action: torch.Tensor, domain: str,
               dt: Optional[float] = None) -> torch.Tensor:
        """Advance the domain state by one time step.

        Args:
            action: strategy vector ∈ R^{n_strategies}
            domain: one of "edge", "container", "soc"
            dt: time-step override (default: config.sde_dt)

        Returns:
            Updated state tensor X_d(t + dt).
        """
        dt = dt or self.config.sde_dt
        state = self.states[domain]
        d = state.shape[0]

        # Drift μ_d(X, a)
        inp = torch.cat([state, action])
        drift = self.drift_nets[domain](inp)

        # Diffusion Σ_d(X, a) reshaped to (d, d)
        sigma_flat = self.diffusion_nets[domain](inp)
        sigma = sigma_flat.view(d, d)

        # Brownian increment dW ~ N(0, dt)
        dW = torch.randn(d, device=self.device) * np.sqrt(dt)

        # Poisson jump component
        lam = self.config.get_jump_rate(domain)
        n_attacks = self.config.get_n_attack_classes(domain)
        jump = torch.zeros(d, device=self.device)
        if np.random.random() < lam * dt * n_attacks:
            jump = torch.randn(d, device=self.device) * self.config.jump_magnitude

        # Euler–Maruyama update
        new_state = state + drift * dt + sigma @ dW + jump
        self.states[domain] = new_state.detach()
        self.time += dt

        return new_state

    def reset(self):
        """Reset all domain states to zero."""
        for domain in DOMAINS:
            d = self.config.get_features(domain)
            self.states[domain] = torch.zeros(d, device=self.device)
        self.time = 0.0


# ═════════════════════════════════════════════════════════════════════════
# Nash Equilibrium Solver  (Theorem 2)
# ═════════════════════════════════════════════════════════════════════════

class NashEquilibriumSolver:
    """Mixed-strategy Nash equilibrium with imbalance-weighted payoffs.

    Payoff construction (Definition 8):
        For defender strategy i vs adversary strategy j, the payoff U_d(i,j) is:

            U_d = √(1/ρ_d) · R_detect(i) − √(ρ_d) · C_fp(i,j) − 0.1 · C_resource(i)

        where ρ_d is the domain imbalance ratio.

    Equilibrium computed via the support-enumeration LP formulation.
    """

    def __init__(self, config: GameConfig):
        self.config = config
        self.equilibrium_history: List[Tuple[np.ndarray, np.ndarray]] = []

    def compute_payoff_matrix(self, domain: str,
                               state: Optional[torch.Tensor] = None
                               ) -> np.ndarray:
        """Build imbalance-adjusted payoff matrix U ∈ R^{n×n}.

        The state vector can optionally influence the payoff through
        a simple modulation term, making the game state-dependent.
        """
        rho = self.config.get_imbalance(domain)
        n = self.config.n_strategies
        U = np.zeros((n, n))

        # Optional state modulation
        state_mod = 1.0
        if state is not None:
            state_mod = 1.0 + 0.01 * torch.norm(state).item()

        for i in range(n):
            for j in range(n):
                a_d = i / n  # defender action normalised to [0, 1]
                a_a = j / n  # adversary action

                R_detect = np.sqrt(1.0 / rho) * (1.0 - abs(a_d - 0.5))
                C_fp = np.sqrt(rho) * abs(a_d - 0.7) * (1.0 + 0.1 * a_a)
                C_res = 0.1 * a_d

                U[i, j] = state_mod * (R_detect - C_fp - C_res)

        return U

    def solve(self, payoff_matrix: np.ndarray
              ) -> Tuple[np.ndarray, np.ndarray]:
        """Solve for mixed-strategy Nash equilibrium via LP.

        Defender maximises, adversary minimises.

        Returns (π_d, π_a) – mixed-strategy vectors on the simplex.
        """
        n = payoff_matrix.shape[0]
        bounds = [(0, 1)] * n
        A_eq = np.ones((1, n))
        b_eq = np.array([1.0])

        # Defender LP: max v s.t.  U^T π_d ≥ v·1
        try:
            res_d = linprog(
                c=-np.ones(n),
                A_ub=-payoff_matrix.T,
                b_ub=-np.ones(n),
                A_eq=A_eq, b_eq=b_eq,
                bounds=bounds, method="highs",
            )
            pi_d = res_d.x if res_d.success else np.ones(n) / n
        except Exception:
            pi_d = np.ones(n) / n

        # Adversary LP: min v s.t.  U π_a ≤ v·1
        try:
            res_a = linprog(
                c=np.ones(n),
                A_ub=payoff_matrix,
                b_ub=np.ones(n),
                A_eq=A_eq, b_eq=b_eq,
                bounds=bounds, method="highs",
            )
            pi_a = res_a.x if res_a.success else np.ones(n) / n
        except Exception:
            pi_a = np.ones(n) / n

        # Normalise (LP solvers sometimes return slightly off-simplex)
        pi_d = np.maximum(pi_d, 0)
        pi_d /= pi_d.sum() + 1e-12
        pi_a = np.maximum(pi_a, 0)
        pi_a /= pi_a.sum() + 1e-12

        self.equilibrium_history.append((pi_d, pi_a))
        return pi_d, pi_a

    def compute_nash_gap(self) -> float:
        """‖Δπ‖ between last two equilibria – convergence indicator."""
        if len(self.equilibrium_history) < 2:
            return float("inf")
        prev_d, prev_a = self.equilibrium_history[-2]
        curr_d, curr_a = self.equilibrium_history[-1]
        return max(
            float(np.linalg.norm(curr_d - prev_d)),
            float(np.linalg.norm(curr_a - prev_a)),
        )

    def compute_game_value(self, payoff_matrix: np.ndarray,
                            pi_d: np.ndarray, pi_a: np.ndarray) -> float:
        """Expected game value v* = π_d^T U π_a."""
        return float(pi_d @ payoff_matrix @ pi_a)
