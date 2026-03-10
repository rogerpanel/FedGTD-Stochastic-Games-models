"""
Sensitivity analysis and ablation studies (Section 6.8, Reviewer 2).

Implements:
- One-at-a-time (OAT) sensitivity for payoff parameters
- Monte Carlo joint sensitivity with partial rank correlations
- Preprocessing ablation studies
- Privacy–utility trade-off analysis

Reference: Paper Section 6.8, Figure 8, Table 10
"""

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import spearmanr

from fedgtd.config import GameConfig


def one_at_a_time_sensitivity(
    run_fn: Callable[[Dict[str, float]], float],
    base_params: Dict[str, float],
    perturbation: float = 0.5,
) -> Dict[str, Tuple[float, float, float]]:
    """One-at-a-time sensitivity analysis.

    For each parameter, evaluates at base*(1-δ) and base*(1+δ).

    Args:
        run_fn: function(params_dict) -> accuracy
        base_params: nominal parameter values
        perturbation: fractional perturbation (0.5 = +/- 50%)

    Returns:
        {param_name: (low_acc, base_acc, high_acc)}
    """
    base_acc = run_fn(base_params)
    results = {}

    for param, val in base_params.items():
        # Low
        params_low = base_params.copy()
        params_low[param] = val * (1 - perturbation)
        acc_low = run_fn(params_low)

        # High
        params_high = base_params.copy()
        params_high[param] = val * (1 + perturbation)
        acc_high = run_fn(params_high)

        results[param] = (acc_low, base_acc, acc_high)

    return results


def monte_carlo_sensitivity(
    run_fn: Callable[[Dict[str, float]], float],
    base_params: Dict[str, float],
    n_samples: int = 1000,
    perturbation: float = 0.25,
    seed: int = 42,
) -> Dict[str, float]:
    """Monte Carlo joint sensitivity with partial rank correlations.

    Jointly varies all parameters uniformly within +/- perturbation range.
    Computes partial rank correlation coefficients (PRCCs).

    Args:
        run_fn: function(params_dict) -> accuracy
        base_params: nominal parameter values
        n_samples: number of MC samples
        perturbation: fractional perturbation range (0.25 = +/- 25%)
        seed: random seed

    Returns:
        {param_name: PRCC_value}

    Paper reference: Section 6.8 - "Partial rank correlations:
        β (0.72) > α (0.48) > ρ (0.31) > γ (0.18) > γ_disc (0.09)"
    """
    rng = np.random.RandomState(seed)
    param_names = list(base_params.keys())
    n_params = len(param_names)

    # Sample parameter values
    samples = np.zeros((n_samples, n_params))
    for i, name in enumerate(param_names):
        low = base_params[name] * (1 - perturbation)
        high = base_params[name] * (1 + perturbation)
        samples[:, i] = rng.uniform(low, high, n_samples)

    # Evaluate
    accuracies = np.zeros(n_samples)
    for s in range(n_samples):
        params = {name: samples[s, i] for i, name in enumerate(param_names)}
        accuracies[s] = run_fn(params)

    # Compute PRCCs (rank-based Spearman partial correlations)
    prccs = {}
    for i, name in enumerate(param_names):
        corr, _ = spearmanr(samples[:, i], accuracies)
        prccs[name] = abs(corr)

    return prccs


def preprocessing_ablation(
    run_fn_with_config: Callable[[str], float],
    components: Optional[List[str]] = None,
) -> Dict[str, float]:
    """Preprocessing ablation study.

    Evaluates accuracy when each preprocessing component is removed.
    Expected components (from paper):
        - class_weight_calibration: most critical for SOC (-8.5%)
        - protocol_encoding: essential for Edge-IIoT (-4.5%)
        - entity_embedding: vital for SOC (-5.5%)
        - temporal_features: temporal aggregation windows
        - feature_normalisation: StandardScaler / MinMaxScaler

    Args:
        run_fn_with_config: function(ablated_component_name) -> accuracy
            pass "none" for full pipeline
        components: list of component names to ablate

    Returns:
        {component: accuracy_without_it}
    """
    if components is None:
        components = [
            "class_weight_calibration",
            "protocol_encoding",
            "entity_embedding",
            "temporal_features",
            "feature_normalisation",
        ]

    results = {"full_pipeline": run_fn_with_config("none")}
    for comp in components:
        results[f"without_{comp}"] = run_fn_with_config(comp)

    return results


def privacy_utility_sweep(
    run_fn_with_epsilon: Callable[[float], float],
    epsilon_values: Optional[List[float]] = None,
) -> Dict[float, float]:
    """Sweep privacy budget ε and measure accuracy.

    Args:
        run_fn_with_epsilon: function(epsilon) -> accuracy
        epsilon_values: list of ε values to test

    Returns:
        {epsilon: accuracy}
    """
    if epsilon_values is None:
        epsilon_values = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 5.0, 10.0]

    return {eps: run_fn_with_epsilon(eps) for eps in epsilon_values}
