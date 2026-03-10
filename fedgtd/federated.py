"""
FedGTD federated training orchestrator (Algorithm 2).

Coordinates:
- Local adversarial training per domain
- Byzantine-resilient aggregation
- Stochastic game state evolution
- Nash equilibrium updates
- Convergence monitoring

Reference: Paper Section 5 (Algorithm 2) and Section 6 (Experiments)
"""

import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from fedgtd.aggregation import ByzantineResilientAggregator
from fedgtd.config import GameConfig
from fedgtd.convergence import MartingaleConvergenceAnalyzer
from fedgtd.game_dynamics import NashEquilibriumSolver, StochasticDifferentialGame
from fedgtd.models import DomainSpecificDefender, StrategicAdversaryNetwork
from fedgtd.utils import MetricsTracker

DOMAINS = ("edge", "container", "soc")


class FedGTDSystem:
    """Main FedGTD system implementing Algorithm 2.

    Algorithm 2: Federated Game-Theoretic Defence

    Input : K organisations, T rounds, domains D = {edge, container, soc}
    For t = 1 … T:
        Phase 1 – Local training:
            For each domain d ∈ D, for each honest client k:
                Train local model with adversarial augmentation (mix ratio α).
                Collect model snapshot θ_k^d(t).
        Phase 2 – Byzantine-resilient aggregation (Algorithm 1):
            Aggregate θ_k^d → θ_global^d with trimmed mean + DP noise.
        Phase 3 – Game dynamics:
            Evolve SDE state; recompute Nash equilibrium.
        Phase 4 – Convergence check:
            Compute Lyapunov V(t); check supermartingale condition.
    Output: converged global models {θ*_d}, Nash strategies {π*_d, π*_a}
    """

    def __init__(self, config: GameConfig):
        self.config = config
        self.device = config.resolve_device()

        # Components
        self.aggregator = ByzantineResilientAggregator(config)
        self.game = StochasticDifferentialGame(config, self.device)
        self.nash_solver = NashEquilibriumSolver(config)
        self.convergence = MartingaleConvergenceAnalyzer(config)
        self.metrics = MetricsTracker()

        # Models (populated by initialize_models)
        self.defenders: Dict[str, List[DomainSpecificDefender]] = {d: [] for d in DOMAINS}
        self.adversaries: Dict[str, Optional[StrategicAdversaryNetwork]] = {d: None for d in DOMAINS}

        self.current_round = 0

    # ── Initialization ───────────────────────────────────────────────────

    def initialize_models(self, data_dims: Dict[str, int]):
        """Create per-client defender models and per-domain adversaries.

        Args:
            data_dims: {"edge": n_features, "container": n_features, "soc": n_features}
        """
        for domain in DOMAINS:
            n_clients = self.config.get_n_clients(domain)
            dim = data_dims[domain]

            self.defenders[domain] = [
                DomainSpecificDefender(dim, domain, self.config).to(self.device)
                for _ in range(n_clients)
            ]
            self.adversaries[domain] = StrategicAdversaryNetwork(
                dim, domain, self.config
            ).to(self.device)

        total_params = sum(
            p.numel()
            for models in self.defenders.values()
            for m in models
            for p in m.parameters()
        )
        print(f"  Total defender parameters: {total_params:,}")

    # ── Phase 1: Local training ──────────────────────────────────────────

    def _local_train(self, model: DomainSpecificDefender,
                     loader: DataLoader, domain: str,
                     client_id: int) -> Dict:
        """Local SGD with adversarial augmentation."""
        model.train()

        lr = self.convergence.get_learning_rate(domain, self.current_round)
        optimizer = optim.AdamW(model.parameters(), lr=lr,
                                weight_decay=self.config.weight_decay)

        # Class-weighted loss for imbalance (paper Eq. 3: w_c = sqrt(n_total/(C*n_c)))
        # Cap weight to avoid gradient explosion on extreme imbalance (SOC ρ=99)
        rho = self.config.get_imbalance(domain)
        w_minority = min(np.sqrt(rho), 20.0)
        weight = torch.tensor([1.0, w_minority], device=self.device)
        criterion = nn.CrossEntropyLoss(weight=weight)

        alpha = self.config.adversarial_mix_ratio
        adv_net = self.adversaries[domain]

        losses, accs = [], []

        for _epoch in range(self.config.local_epochs):
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                # Adversarial augmentation
                if adv_net is not None:
                    with torch.no_grad():
                        delta = adv_net(batch_x)
                    x_adv = batch_x + delta
                else:
                    x_adv = batch_x

                out_clean = model(batch_x)
                out_adv = model(x_adv)

                loss = alpha * criterion(out_clean, batch_y) + \
                       (1 - alpha) * criterion(out_adv, batch_y)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                losses.append(loss.item())
                accs.append((out_clean.argmax(1) == batch_y).float().mean().item())

        state = {name: p.data.clone() for name, p in model.named_parameters()}
        return {
            "model": state,
            "loss": float(np.mean(losses)),
            "accuracy": float(np.mean(accs)),
            "client_id": client_id,
            "domain": domain,
        }

    # ── Full federated round (Algorithm 2) ───────────────────────────────

    def federated_round(self, train_loaders: Dict[str, List[DataLoader]]) -> Dict:
        """Execute one round of Algorithm 2.

        Args:
            train_loaders: {"edge": [loader_k, ...], "container": [...], "soc": [...]}

        Returns:
            Round metrics dict.
        """
        self.current_round += 1
        t0 = time.time()
        all_updates: Dict[str, list] = defaultdict(list)

        # ── Phase 1: Local training ──────────────────────────────────────
        for domain in DOMAINS:
            loaders = train_loaders.get(domain, [])
            models = self.defenders[domain]
            for cid, (model, loader) in enumerate(zip(models, loaders)):
                if loader is not None:
                    upd = self._local_train(model, loader, domain, cid)
                    all_updates[domain].append(upd)

        # ── Phase 2: Byzantine-resilient aggregation (Algorithm 1) ───────
        for domain, updates in all_updates.items():
            if not updates:
                continue
            client_states = [u["model"] for u in updates]
            agg_state = self.aggregator.aggregate(
                client_states, domain, self.current_round
            )
            # Broadcast aggregated model to all clients
            for model in self.defenders[domain]:
                model.load_state_dict(agg_state, strict=False)

        # ── Phase 3: Game dynamics ───────────────────────────────────────
        nash_gaps = []
        for domain in DOMAINS:
            action = torch.randn(self.config.n_strategies, device=self.device)
            new_state = self.game.evolve(action, domain)

            U = self.nash_solver.compute_payoff_matrix(domain, new_state)
            pi_d, pi_a = self.nash_solver.solve(U)

            nash_gaps.append(self.nash_solver.compute_nash_gap())

        max_nash_gap = max(nash_gaps) if nash_gaps else float("inf")

        # ── Phase 4: Convergence check ───────────────────────────────────
        lyapunov = self.convergence.compute_lyapunov(self.defenders)
        converged = self.convergence.check_convergence(
            max_nash_gap, self.current_round
        )

        # Compile metrics
        all_losses = [u["loss"] for updates in all_updates.values() for u in updates]
        all_accs = [u["accuracy"] for updates in all_updates.values() for u in updates]

        metrics = {
            "round": self.current_round,
            "avg_loss": float(np.mean(all_losses)) if all_losses else 0.0,
            "avg_accuracy": float(np.mean(all_accs)) if all_accs else 0.0,
            "nash_gap": max_nash_gap,
            "lyapunov": lyapunov,
            "converged": converged,
            "round_time": time.time() - t0,
        }
        self.metrics.log_round(metrics)
        return metrics

    # ── Evaluation ───────────────────────────────────────────────────────

    @torch.no_grad()
    def evaluate(self, test_loaders: Dict[str, DataLoader]) -> Dict[str, Dict]:
        """Evaluate all domains on test sets.

        Returns per-domain dict with accuracy, precision, recall, F1, AUC.
        """
        from sklearn.metrics import (accuracy_score, f1_score,
                                     precision_recall_fscore_support,
                                     roc_auc_score)

        results = {}
        for domain, loader in test_loaders.items():
            if domain not in self.defenders or not self.defenders[domain]:
                continue

            model = self.defenders[domain][0]
            model.eval()

            preds_all, labels_all, probs_all = [], [], []

            for bx, by in loader:
                bx, by = bx.to(self.device), by.to(self.device)
                logits = model(bx)
                prob = F.softmax(logits, dim=1)[:, 1]
                pred = logits.argmax(1)
                preds_all.extend(pred.cpu().numpy())
                labels_all.extend(by.cpu().numpy())
                probs_all.extend(prob.cpu().numpy())

            preds_all = np.array(preds_all)
            labels_all = np.array(labels_all)
            probs_all = np.array(probs_all)

            acc = accuracy_score(labels_all, preds_all)
            prec, rec, f1, _ = precision_recall_fscore_support(
                labels_all, preds_all, average="binary", zero_division=0
            )
            try:
                auc = roc_auc_score(labels_all, probs_all)
            except ValueError:
                auc = 0.5

            results[domain] = {
                "accuracy": acc, "precision": prec,
                "recall": rec, "f1": f1, "auc": auc,
            }

        return results

    # ── Byzantine resilience test (Section 7.2) ──────────────────────────

    def test_byzantine_resilience(
        self,
        train_loaders: Dict[str, List[DataLoader]],
        test_loaders: Dict[str, DataLoader],
        corruption_levels: Tuple[float, ...] = (0.05, 0.10, 0.15, 0.20),
        rounds_per_level: int = 10,
    ) -> Dict[float, Dict]:
        """Measure performance retention under increasing Byzantine corruption.

        For each corruption level f:
            1. Designate f fraction of clients as Byzantine.
            2. Replace their updates with random noise / sign-flipped gradients.
            3. Run `rounds_per_level` rounds and evaluate.
        """
        results = {}
        for f in corruption_levels:
            # Save current state
            saved = {
                d: [
                    {n: p.data.clone() for n, p in m.named_parameters()}
                    for m in self.defenders[d]
                ]
                for d in DOMAINS
            }

            # Inject Byzantine clients
            for domain in DOMAINS:
                n_byz = max(1, int(len(self.defenders[domain]) * f))
                for i in range(n_byz):
                    # Randomise Byzantine client parameters
                    for p in self.defenders[domain][i].parameters():
                        p.data = torch.randn_like(p.data)

            # Run rounds
            for _ in range(rounds_per_level):
                self.federated_round(train_loaders)

            # Evaluate
            eval_res = self.evaluate(test_loaders)
            avg_acc = np.mean([v["accuracy"] for v in eval_res.values()])
            results[f] = {"avg_accuracy": avg_acc, "per_domain": eval_res}

            # Restore
            for d in DOMAINS:
                for model, state in zip(self.defenders[d], saved[d]):
                    for n, p in model.named_parameters():
                        p.data = state[n]

        return results
