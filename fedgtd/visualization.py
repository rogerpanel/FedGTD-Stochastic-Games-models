"""
Visualization module for FedGTD experiments.

Generates all figures aligned with the paper:
- Fig 1: Convergence curves (accuracy, loss) per domain
- Fig 2: Nash gap evolution (log scale)
- Fig 3: Lyapunov stability trajectory
- Fig 4: Byzantine resilience bar chart
- Fig 5: Adversarial robustness curves
- Fig 6: Baseline comparison radar / bar chart
- Fig 7: Confusion matrices per domain
- Fig 8: Privacy–utility trade-off

Reference: Paper Figures 1–8 and Supplementary Figures
"""

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Publication-quality defaults
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 10,
    "figure.figsize": (7, 5),
})


def plot_convergence(round_metrics: List[dict], save_dir: Optional[Path] = None):
    """Fig 1 + 2 + 3: Accuracy, loss, Nash gap, Lyapunov."""
    rounds = [m["round"] for m in round_metrics]
    accs = [m["avg_accuracy"] for m in round_metrics]
    losses = [m["avg_loss"] for m in round_metrics]
    nash_gaps = [m["nash_gap"] for m in round_metrics]
    lyap = [m["lyapunov"] for m in round_metrics]

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    # Accuracy
    axes[0, 0].plot(rounds, accs, "b-", linewidth=2, label="FedGTD")
    axes[0, 0].set_xlabel("Round")
    axes[0, 0].set_ylabel("Accuracy")
    axes[0, 0].set_title("(a) Federated Convergence")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()

    # Loss
    axes[0, 1].plot(rounds, losses, "r-", linewidth=2)
    axes[0, 1].set_xlabel("Round")
    axes[0, 1].set_ylabel("Loss")
    axes[0, 1].set_title("(b) Training Loss")
    axes[0, 1].grid(True, alpha=0.3)

    # Nash gap (log)
    pos_gaps = [(r, g) for r, g in zip(rounds, nash_gaps) if g > 0 and np.isfinite(g)]
    if pos_gaps:
        rr, gg = zip(*pos_gaps)
        axes[1, 0].semilogy(rr, gg, "g-", linewidth=2)
    axes[1, 0].set_xlabel("Round")
    axes[1, 0].set_ylabel("Nash Gap (log)")
    axes[1, 0].set_title("(c) Nash Equilibrium Convergence")
    axes[1, 0].grid(True, alpha=0.3)

    # Lyapunov
    axes[1, 1].plot(rounds, lyap, color="purple", linewidth=2)
    axes[1, 1].set_xlabel("Round")
    axes[1, 1].set_ylabel("V(t)")
    axes[1, 1].set_title("(d) Lyapunov Stability")
    axes[1, 1].grid(True, alpha=0.3)

    fig.suptitle("FedGTD Performance Analysis", fontsize=14, y=1.01)
    plt.tight_layout()
    if save_dir:
        fig.savefig(save_dir / "convergence.png", bbox_inches="tight")
    plt.show()


def plot_byzantine_resilience(resilience_results: Dict[float, Dict],
                               save_dir: Optional[Path] = None):
    """Fig 4: Byzantine resilience – performance retention bar chart."""
    levels = sorted(resilience_results.keys())
    accs = [resilience_results[f]["avg_accuracy"] for f in levels]
    labels = [f"{int(f*100)}%" for f in levels]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, [a * 100 for a in accs],
                  color=sns.color_palette("viridis", len(levels)), edgecolor="black")
    ax.set_xlabel("Byzantine Corruption Level")
    ax.set_ylabel("Average Accuracy (%)")
    ax.set_title("Byzantine Resilience – Performance Retention")
    ax.set_ylim(0, 105)
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{acc*100:.1f}%", ha="center", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    if save_dir:
        fig.savefig(save_dir / "byzantine_resilience.png", bbox_inches="tight")
    plt.show()


def plot_adversarial_robustness(adv_results: Dict[str, Dict[str, Dict]],
                                 save_dir: Optional[Path] = None):
    """Fig 5: Adversarial robustness curves per domain."""
    fig, axes = plt.subplots(1, len(adv_results), figsize=(5 * len(adv_results), 5),
                              squeeze=False)

    for idx, (domain, eps_dict) in enumerate(adv_results.items()):
        ax = axes[0, idx]
        epsilons = []
        clean_accs, fgsm_accs, pgd_accs, cw_accs = [], [], [], []

        for key in sorted(eps_dict.keys()):
            eps_val = float(key.split("_")[1])
            epsilons.append(eps_val)
            clean_accs.append(eps_dict[key]["clean"] * 100)
            fgsm_accs.append(eps_dict[key]["fgsm"] * 100)
            pgd_accs.append(eps_dict[key]["pgd"] * 100)
            cw_accs.append(eps_dict[key].get("cw", eps_dict[key]["pgd"]) * 100)

        ax.plot(epsilons, clean_accs, "o-", label="Clean", linewidth=2)
        ax.plot(epsilons, fgsm_accs, "s--", label="FGSM", linewidth=2)
        ax.plot(epsilons, pgd_accs, "^:", label="PGD", linewidth=2)
        ax.plot(epsilons, cw_accs, "d-.", label="C&W", linewidth=2)
        ax.set_xlabel("Perturbation ε")
        ax.set_ylabel("Accuracy (%)")
        ax.set_title(f"{domain.upper()} Domain")
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle("Adversarial Robustness Analysis", fontsize=14, y=1.02)
    plt.tight_layout()
    if save_dir:
        fig.savefig(save_dir / "adversarial_robustness.png", bbox_inches="tight")
    plt.show()


def plot_baseline_comparison(fedgtd_results: Dict[str, Dict],
                              baseline_results: Dict[str, Dict[str, Dict]],
                              save_dir: Optional[Path] = None):
    """Fig 6: Grouped bar chart – FedGTD vs baselines per domain."""
    domains = list(fedgtd_results.keys())
    methods = ["FedGTD"] + list(baseline_results.keys())
    n_methods = len(methods)
    x = np.arange(len(domains))
    width = 0.8 / n_methods

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = sns.color_palette("Set2", n_methods)

    for i, method in enumerate(methods):
        accs = []
        for d in domains:
            if method == "FedGTD":
                accs.append(fedgtd_results[d]["accuracy"] * 100)
            elif d in baseline_results.get(method, {}):
                accs.append(baseline_results[method][d]["accuracy"] * 100)
            else:
                accs.append(0)
        offset = (i - n_methods / 2 + 0.5) * width
        ax.bar(x + offset, accs, width, label=method, color=colors[i],
               edgecolor="black", linewidth=0.5)

    ax.set_xlabel("Domain")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("FedGTD vs Baseline Methods")
    ax.set_xticks(x)
    ax.set_xticklabels([d.upper() for d in domains])
    ax.legend(loc="lower right")
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    if save_dir:
        fig.savefig(save_dir / "baseline_comparison.png", bbox_inches="tight")
    plt.show()


def plot_confusion_matrices(predictions: Dict[str, Dict],
                             save_dir: Optional[Path] = None):
    """Fig 7: Per-domain confusion matrices."""
    from sklearn.metrics import confusion_matrix

    n_domains = len(predictions)
    fig, axes = plt.subplots(1, n_domains, figsize=(5 * n_domains, 4.5))
    if n_domains == 1:
        axes = [axes]

    for idx, (domain, data) in enumerate(predictions.items()):
        cm = confusion_matrix(data["labels"], data["preds"])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[idx],
                    xticklabels=["Benign", "Attack"],
                    yticklabels=["Benign", "Attack"])
        axes[idx].set_title(f"{domain.upper()}")
        axes[idx].set_xlabel("Predicted")
        axes[idx].set_ylabel("True")

    fig.suptitle("Confusion Matrices", fontsize=14, y=1.02)
    plt.tight_layout()
    if save_dir:
        fig.savefig(save_dir / "confusion_matrices.png", bbox_inches="tight")
    plt.show()


def plot_privacy_utility(privacy_results: Dict[float, float],
                          save_dir: Optional[Path] = None):
    """Fig 8: Privacy budget (ε) vs utility (accuracy) trade-off."""
    epsilons = sorted(privacy_results.keys())
    accuracies = [privacy_results[e] * 100 for e in epsilons]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(epsilons, accuracies, "o-", color="teal", linewidth=2, markersize=8)
    ax.set_xlabel("Privacy Budget ε")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Privacy–Utility Trade-off")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_dir:
        fig.savefig(save_dir / "privacy_utility.png", bbox_inches="tight")
    plt.show()
