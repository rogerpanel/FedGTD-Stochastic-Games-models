# FedGTD: Byzantine-Resilient Stochastic Games for Federated Multi-Cloud Intrusion Detection

**Reproducibility package** for the paper:

> Anaedevha et al., "Byzantine-Resilient Stochastic Games for Federated Multi-Cloud Intrusion Detection",
> *Complex and Intelligent Systems* (2026).

---

## Overview

This codebase implements the complete **FedGTD** framework — a federated multi-cloud intrusion detection system that combines stochastic game theory with Byzantine-resilient aggregation. The system operates across three heterogeneous cloud security domains:

| Domain | Dataset | Samples | Features | Attack Classes | Imbalance (ρ) |
|--------|---------|---------|----------|----------------|---------------|
| **Edge-IIoT** | Edge-IIoT (Ferrag et al.) | 2,219,201 | 60–140 | 14 families | 2.67 |
| **Container** | Container (Caprolu et al.) | 234,560 | 87 | 11 CVEs | 15.7 |
| **SOC** | Microsoft GUIDE | 13M+ | 46 | 33 entities | 99.0 |

### Key Components

1. **Stochastic Differential Game** (Section 4.1.2): SDE-based state evolution with Poisson jump-diffusion modelling attack arrivals
2. **Nash Equilibrium Solver** (Theorem 2): Imbalance-weighted payoff matrices with LP-based mixed-strategy computation
3. **Byzantine-Resilient Aggregation** (Algorithm 1): Cross-domain projection, cosine-similarity detection, trimmed mean with DP noise
4. **Federated Training Orchestrator** (Algorithm 2): Domain-adaptive learning rates, adversarial augmentation, convergence monitoring
5. **Martingale Convergence Analysis** (Theorem 4): Heterogeneous Lyapunov function with supermartingale convergence guarantee

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

Or install as a package:
```bash
pip install -e .
```

### 2. Configure Kaggle (for real data)

```bash
# Set up Kaggle API credentials
export KAGGLE_USERNAME=your_username
export KAGGLE_KEY=your_api_key
```

The ICS3D dataset will be auto-downloaded from:
**https://doi.org/10.34740/kaggle/dsv/12483891**

If Kaggle credentials are not configured, the code automatically falls back to synthetic data that mirrors the statistical properties of the real datasets.

### 3. Run Experiments

```bash
# Quick demo (CPU OK, ~10 minutes)
python run_experiments.py --demo

# Full experiment (GPU recommended, ~2-4 hours)
python run_experiments.py

# Custom configuration
python run_experiments.py --rounds 100 --device cuda --max-samples 50000

# Skip specific analyses for faster runs
python run_experiments.py --skip-baselines --skip-adversarial
```

---

## Project Structure

```
fedgtd_byzantine/
├── README.md                  # This file
├── requirements.txt           # Python dependencies
├── setup.py                   # Package installer
├── run_experiments.py         # Main experiment entry point
│
├── fedgtd/                    # Core package
│   ├── __init__.py
│   ├── config.py              # All hyperparameters (Table 2)
│   ├── datasets.py            # ICS3D data loading & federated partitioning
│   ├── models.py              # Defender & adversary neural architectures
│   ├── aggregation.py         # Algorithm 1: Byzantine-resilient aggregation
│   ├── game_dynamics.py       # SDE game dynamics & Nash equilibrium solver
│   ├── convergence.py         # Theorem 4: Martingale convergence analysis
│   ├── federated.py           # Algorithm 2: FedGTD training orchestrator
│   ├── adversarial.py         # FGSM, PGD, C&W robustness evaluation
│   ├── baselines.py           # FedAvg, FedProx, Krum, TrimmedMean baselines
│   ├── visualization.py       # Publication-quality figure generation
│   └── utils.py               # Seeding, metrics tracking, timing
│
├── configs/                   # Optional YAML/JSON config overrides
└── results/                   # Output directory (created at runtime)
    ├── results_summary.json   # Key metrics in JSON
    ├── full_metrics.pkl       # Complete metrics (pickle)
    └── figures/               # Generated plots
        ├── convergence.png
        ├── byzantine_resilience.png
        ├── adversarial_robustness.png
        └── baseline_comparison.png
```

---

## Paper–Code Correspondence

| Paper Section | Code Module | Description |
|---------------|-------------|-------------|
| Section 3 (Definitions 5–6) | `config.py` | Federation topology, privacy budgets |
| Section 3.2 | `models.py:StrategicAdversaryNetwork` | Adversary model |
| Section 4.1.2 (Eqs. 7–9) | `game_dynamics.py:StochasticDifferentialGame` | SDE state evolution |
| Section 4.3 (Theorem 2) | `game_dynamics.py:NashEquilibriumSolver` | Nash equilibrium |
| Section 4.4 | `models.py:DomainSpecificDefender` | Neural architectures |
| Section 5 (Algorithm 1) | `aggregation.py:ByzantineResilientAggregator` | Byzantine detection + aggregation |
| Section 5 (Algorithm 2) | `federated.py:FedGTDSystem` | Full federated training |
| Theorem 3 | `aggregation.py:add_dp_noise` | Differential privacy |
| Theorem 4 (Eq. 12) | `convergence.py:MartingaleConvergenceAnalyzer` | Lyapunov convergence |
| Section 6.3 | `datasets.py:create_federated_splits` | Dirichlet non-IID partition |
| Section 7.1 (Table 3) | `baselines.py` | Baseline comparisons |
| Section 7.3 | `adversarial.py:AdversarialEvaluator` | Adversarial robustness |

---

## Configuration

All hyperparameters are centralised in `fedgtd/config.py` as a `GameConfig` dataclass. Key parameters:

| Parameter | Value | Paper Reference |
|-----------|-------|-----------------|
| K (organisations) | 20 | Section 3.1 |
| T (max rounds) | 200 | Section 6.4 |
| E (local epochs) | 5 | Table 2 |
| γ (discount factor) | 0.95 | Section 4 |
| α (Dirichlet) | 0.3 | Section 6.3 |
| ε_edge / δ_edge | 2.5 / 1e-5 | Definition 6 |
| C_edge (clip norm) | 0.61 | Section 6.4 |
| f (Byzantine fraction) | 0.15 | Section 5 |

---

## Reproducibility Notes

- **Random seeds**: All sources of randomness (NumPy, PyTorch, CUDA) are seeded via `set_seeds(42)`.
- **Deterministic mode**: CUDA deterministic mode is enabled (`cudnn.deterministic = True`).
- **Data splits**: Stratified 80/20 train/test split with `random_state=42`.
- **Federated partition**: Dirichlet(α=0.3) ensures consistent non-IID distribution across runs.

---

## Citation

```bibtex
@article{anaedevha2026byzantine,
  title={Byzantine-Resilient Stochastic Games for Federated Multi-Cloud Intrusion Detection},
  author={Anaedevha, Roger Nick and others},
  journal={Complex and Intelligent Systems},
  year={2026},
  publisher={Springer}
}
```

## Dataset

```bibtex
@misc{anaedevha2025ics3d,
  title={Integrated Cloud Security 3-Datasets (ICS3D)},
  author={Anaedevha, Roger Nick},
  year={2025},
  doi={10.34740/kaggle/dsv/12483891},
  publisher={Kaggle}
}
```
