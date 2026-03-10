#!/usr/bin/env python3
"""
FedGTD: Byzantine-Resilient Stochastic Games for Federated Multi-Cloud
        Intrusion Detection – Main Experiment Runner

This script reproduces all experiments from:
    Anaedevha et al., "Byzantine-Resilient Stochastic Games for Federated
    Multi-Cloud Intrusion Detection", Complex and Intelligent Systems (2026).

Usage:
    # Full experiments (requires GPU, ~2-4 hours)
    python run_experiments.py

    # Quick demo with synthetic data (CPU OK, ~10 min)
    python run_experiments.py --demo

    # Custom settings
    python run_experiments.py --rounds 100 --device cuda --max-samples 50000

Dataset:
    ICS3D – https://doi.org/10.34740/kaggle/dsv/12483891
    Auto-downloaded via kagglehub if Kaggle credentials are configured.
    Falls back to synthetic data otherwise.
"""

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch

from fedgtd.adversarial import AdversarialEvaluator
from fedgtd.baselines import run_all_baselines
from fedgtd.config import GameConfig
from fedgtd.datasets import ICS3DDataHandler
from fedgtd.federated import FedGTDSystem
from fedgtd.utils import MetricsTracker, Timer, set_seeds
from fedgtd.visualization import (
    plot_adversarial_robustness,
    plot_baseline_comparison,
    plot_byzantine_resilience,
    plot_convergence,
)

DOMAINS = ("edge", "container", "soc")


def parse_args():
    parser = argparse.ArgumentParser(
        description="FedGTD Experiment Runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--demo", action="store_true",
                        help="Quick demo mode (fewer rounds, synthetic data)")
    parser.add_argument("--rounds", type=int, default=None,
                        help="Override max federated rounds")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cuda", "cpu"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Cap per-dataset samples (useful for quick runs)")
    parser.add_argument("--output-dir", type=str, default="results",
                        help="Directory for output files")
    parser.add_argument("--skip-baselines", action="store_true",
                        help="Skip baseline comparisons (faster)")
    parser.add_argument("--skip-adversarial", action="store_true",
                        help="Skip adversarial robustness tests")
    parser.add_argument("--skip-byzantine-test", action="store_true",
                        help="Skip Byzantine resilience test")
    parser.add_argument("--baseline-rounds", type=int, default=30,
                        help="Rounds for baseline methods")
    return parser.parse_args()


def main():
    args = parse_args()

    # ── Setup ────────────────────────────────────────────────────────────
    config = GameConfig(seed=args.seed, device=args.device)
    device = config.resolve_device()

    if args.demo:
        config.max_rounds = 20
        config.local_epochs = 2
        max_samples = args.max_samples or 5_000
    else:
        if args.rounds:
            config.max_rounds = args.rounds
        max_samples = args.max_samples

    set_seeds(config.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    print("=" * 72)
    print("  FEDGTD: BYZANTINE-RESILIENT STOCHASTIC GAMES FOR")
    print("  FEDERATED MULTI-CLOUD INTRUSION DETECTION")
    print("=" * 72)
    print(f"  Device     : {device}")
    print(f"  Seed       : {config.seed}")
    print(f"  Max rounds : {config.max_rounds}")
    print(f"  Demo mode  : {args.demo}")
    print(f"  Output     : {output_dir.resolve()}")
    print("=" * 72)

    # ── 1. Load Data ─────────────────────────────────────────────────────
    print("\n[1/7] Loading ICS3D datasets...")
    handler = ICS3DDataHandler(config)
    data_path = handler.download()

    with Timer("Edge-IIoT"):
        X_edge, y_edge = handler.load_edge_iiot(data_path, max_samples)
    with Timer("Container"):
        X_cont, y_cont = handler.load_container(data_path, max_samples)
    with Timer("SOC/GUIDE"):
        X_soc, y_soc = handler.load_soc(data_path, max_samples)

    # ── 2. Train/Test Split & Federated Partition ────────────────────────
    print("\n[2/7] Splitting data (80/20) and creating federated partitions...")
    from sklearn.model_selection import train_test_split

    splits = {}
    for name, X, y in [("edge", X_edge, y_edge),
                        ("container", X_cont, y_cont),
                        ("soc", X_soc, y_soc)]:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, random_state=config.seed, stratify=y
        )
        fed_splits = handler.create_federated_splits(
            X_tr, y_tr, config.get_n_clients(name), config.dirichlet_alpha
        )
        splits[name] = {
            "train_splits": fed_splits,
            "X_test": X_te, "y_test": y_te,
            "n_features": X_tr.shape[1],
        }
        print(f"  {name.upper()}: {len(fed_splits)} clients, "
              f"test size = {len(X_te):,}")

    data_dims = {d: splits[d]["n_features"] for d in DOMAINS}

    # Create data loaders
    train_loaders = {}
    test_loaders = {}
    for d in DOMAINS:
        bs = config.get_batch_size(d)
        train_loaders[d] = handler.splits_to_loaders(splits[d]["train_splits"], bs)
        test_loaders[d] = handler.arrays_to_loader(
            splits[d]["X_test"], splits[d]["y_test"], bs
        )

    # ── 3. Initialize FedGTD ─────────────────────────────────────────────
    print("\n[3/7] Initializing FedGTD system...")
    system = FedGTDSystem(config)
    system.initialize_models(data_dims)

    # ── 4. Federated Training (Algorithm 2) ──────────────────────────────
    print(f"\n[4/7] Federated training ({config.max_rounds} rounds)...")
    print("-" * 56)

    convergence_round = None
    for r in range(1, config.max_rounds + 1):
        metrics = system.federated_round(train_loaders)

        if r % 5 == 0 or r == 1:
            print(f"  Round {r:>3}/{config.max_rounds}  |  "
                  f"loss={metrics['avg_loss']:.4f}  "
                  f"acc={metrics['avg_accuracy']:.4f}  "
                  f"Nash={metrics['nash_gap']:.6f}  "
                  f"V(t)={metrics['lyapunov']:.4f}  "
                  f"({metrics['round_time']:.1f}s)")

        if metrics["converged"] and convergence_round is None:
            convergence_round = r
            print(f"\n  >> Converged at round {r}!")
            break

    # ── 5. Evaluation ────────────────────────────────────────────────────
    print("\n[5/7] Evaluating final models...")
    print("-" * 56)
    fedgtd_results = system.evaluate(test_loaders)

    print(f"\n  {'Domain':<12} {'Acc':>8} {'Prec':>8} {'Rec':>8} "
          f"{'F1':>8} {'AUC':>8}")
    print("  " + "-" * 52)
    for d, m in fedgtd_results.items():
        print(f"  {d.upper():<12} {m['accuracy']*100:>7.1f}% "
              f"{m['precision']*100:>7.1f}% {m['recall']*100:>7.1f}% "
              f"{m['f1']*100:>7.1f}% {m['auc']:.3f}")

    # ── 6. Additional analyses ───────────────────────────────────────────
    print("\n[6/7] Running additional analyses...")

    # 6a. Byzantine resilience test
    byz_results = {}
    if not args.skip_byzantine_test:
        print("\n  [6a] Byzantine resilience test...")
        byz_results = system.test_byzantine_resilience(
            train_loaders, test_loaders,
            corruption_levels=(0.05, 0.10, 0.15, 0.20),
            rounds_per_level=5 if args.demo else 10,
        )
        for f, res in sorted(byz_results.items()):
            print(f"    {int(f*100):>2}% corruption: "
                  f"avg accuracy = {res['avg_accuracy']*100:.1f}%")

    # 6b. Adversarial robustness
    adv_results = {}
    if not args.skip_adversarial:
        print("\n  [6b] Adversarial robustness evaluation...")
        epsilons = [0.01, 0.05, 0.1, 0.2]
        for d in DOMAINS:
            if system.defenders[d]:
                evaluator = AdversarialEvaluator(system.defenders[d][0], device)
                adv_results[d] = evaluator.evaluate(
                    test_loaders[d], epsilons,
                    max_samples=500 if args.demo else 2000,
                )
                for ek, ev in adv_results[d].items():
                    eps_v = ek.split("_")[1]
                    print(f"    {d.upper()} eps={eps_v}: "
                          f"clean={ev['clean']:.3f} "
                          f"FGSM={ev['fgsm']:.3f} "
                          f"PGD={ev['pgd']:.3f} "
                          f"C&W={ev['cw']:.3f}")

    # 6c. Baseline comparisons
    baseline_results = {}
    if not args.skip_baselines:
        print("\n  [6c] Running baseline comparisons...")
        baseline_results = run_all_baselines(
            config, train_loaders, test_loaders, data_dims,
            n_rounds=args.baseline_rounds if not args.demo else 10,
        )
        for method, domain_res in baseline_results.items():
            avg_acc = np.mean([v["accuracy"] for v in domain_res.values()])
            print(f"    {method:<15} avg accuracy = {avg_acc*100:.1f}%")

    # ── 7. Save & Visualize ──────────────────────────────────────────────
    print("\n[7/7] Saving results and generating figures...")

    # Communication efficiency stats
    total_params = sum(
        p.numel() for models in system.defenders.values()
        for m in models for p in m.parameters()
    )
    comm_per_round_mb = total_params * 4 / 1024 / 1024
    final_round = convergence_round or config.max_rounds
    total_comm_gb = comm_per_round_mb * final_round / 1024

    summary = {
        "fedgtd_results": {d: {k: float(v) for k, v in m.items()}
                           for d, m in fedgtd_results.items()},
        "convergence_round": convergence_round,
        "total_rounds": final_round,
        "total_params": total_params,
        "comm_per_round_mb": comm_per_round_mb,
        "total_comm_gb": total_comm_gb,
        "config": {k: v for k, v in vars(config).items()
                   if not k.startswith("_")},
    }

    with open(output_dir / "results_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    with open(output_dir / "full_metrics.pkl", "wb") as f:
        pickle.dump({
            "round_metrics": system.metrics.round_metrics,
            "fedgtd_results": fedgtd_results,
            "byzantine_results": byz_results,
            "adversarial_results": adv_results,
            "baseline_results": baseline_results,
        }, f)

    # Plots
    try:
        if system.metrics.round_metrics:
            plot_convergence(system.metrics.round_metrics, fig_dir)
        if byz_results:
            plot_byzantine_resilience(byz_results, fig_dir)
        if adv_results:
            plot_adversarial_robustness(adv_results, fig_dir)
        if baseline_results:
            plot_baseline_comparison(fedgtd_results, baseline_results, fig_dir)
    except Exception as e:
        print(f"  Warning: visualization error ({e}). Results saved to files.")

    # Final summary
    print("\n" + "=" * 72)
    print("  EXPERIMENT COMPLETE")
    print("=" * 72)
    print(f"  Total parameters     : {total_params:,}")
    print(f"  Comm / round         : {comm_per_round_mb:.2f} MB")
    print(f"  Total communication  : {total_comm_gb:.4f} GB")
    print(f"  Convergence round    : {convergence_round or 'N/A'}")

    for d, m in fedgtd_results.items():
        print(f"  {d.upper():<12} Acc={m['accuracy']*100:.1f}%  "
              f"F1={m['f1']*100:.1f}%  AUC={m['auc']:.3f}")

    print(f"\n  Results saved to: {output_dir.resolve()}/")
    print("=" * 72)


if __name__ == "__main__":
    main()
