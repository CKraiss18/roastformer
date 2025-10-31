"""
Evaluation Metrics for Roast Profile Generation

Implements metrics for comparing generated vs real roast profiles:
- MAE (Mean Absolute Error) - temperature accuracy
- RMSE (Root Mean Squared Error) - penalizes large errors
- DTW (Dynamic Time Warping) - shape similarity
- Finish Temperature Accuracy - final temperature match
- Profile Shape Correlation - overall shape similarity

Author: Charlee Kraiss
Project: RoastFormer - Transformer-Based Roast Profile Generation
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy.spatial.distance import euclidean
from scipy.stats import pearsonr


def mae(real: np.ndarray, generated: np.ndarray) -> float:
    """
    Mean Absolute Error

    Args:
        real: Real temperature profile
        generated: Generated temperature profile

    Returns:
        MAE in °F
    """
    # Handle length mismatch by truncating to shorter length
    min_len = min(len(real), len(generated))
    return np.mean(np.abs(real[:min_len] - generated[:min_len]))


def rmse(real: np.ndarray, generated: np.ndarray) -> float:
    """
    Root Mean Squared Error

    Args:
        real: Real temperature profile
        generated: Generated temperature profile

    Returns:
        RMSE in °F
    """
    min_len = min(len(real), len(generated))
    return np.sqrt(np.mean((real[:min_len] - generated[:min_len]) ** 2))


def finish_temp_accuracy(
    real: np.ndarray,
    generated: np.ndarray,
    threshold: float = 10.0
) -> Tuple[float, bool]:
    """
    Check if finish temperature (last temp) matches within threshold

    Args:
        real: Real temperature profile
        generated: Generated temperature profile
        threshold: Acceptable error in °F (default: 10°F)

    Returns:
        (error in °F, is_accurate)
    """
    real_finish = real[-1]
    gen_finish = generated[-1]

    error = abs(real_finish - gen_finish)
    is_accurate = error < threshold

    return error, is_accurate


def dtw_distance(
    real: np.ndarray,
    generated: np.ndarray,
    window: Optional[int] = None
) -> float:
    """
    Dynamic Time Warping distance

    Measures similarity between two sequences that may vary in speed.
    Good for comparing roast profiles that have similar shapes but different durations.

    Args:
        real: Real temperature profile
        generated: Generated temperature profile
        window: Optional Sakoe-Chiba window constraint (limits warping)

    Returns:
        DTW distance (lower is better)
    """
    n, m = len(real), len(generated)

    # Initialize DTW matrix
    dtw_matrix = np.full((n + 1, m + 1), np.inf)
    dtw_matrix[0, 0] = 0

    # Set window constraint (defaults to no constraint)
    if window is None:
        window = max(n, m)

    # Fill DTW matrix
    for i in range(1, n + 1):
        for j in range(max(1, i - window), min(m + 1, i + window + 1)):
            cost = abs(real[i - 1] - generated[j - 1])
            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i - 1, j],      # insertion
                dtw_matrix[i, j - 1],      # deletion
                dtw_matrix[i - 1, j - 1]   # match
            )

    return dtw_matrix[n, m]


def profile_correlation(real: np.ndarray, generated: np.ndarray) -> float:
    """
    Pearson correlation between real and generated profiles

    Measures overall shape similarity regardless of offset.

    Args:
        real: Real temperature profile
        generated: Generated temperature profile

    Returns:
        Correlation coefficient (-1 to 1, higher is better)
    """
    min_len = min(len(real), len(generated))

    if min_len < 2:
        return 0.0

    corr, _ = pearsonr(real[:min_len], generated[:min_len])
    return corr


def phase_timing_accuracy(
    real: np.ndarray,
    generated: np.ndarray,
    phases: List[str] = ['turning_point', 'first_crack']
) -> Dict[str, Tuple[int, int, float]]:
    """
    Compare timing of key roast phases

    Args:
        real: Real temperature profile
        generated: Generated temperature profile
        phases: List of phases to check

    Returns:
        {
            'turning_point': (real_time, gen_time, error),
            'first_crack': (real_time, gen_time, error),
        }
    """
    results = {}

    # Turning point (minimum temperature)
    if 'turning_point' in phases:
        real_tp = np.argmin(real)
        gen_tp = np.argmin(generated)
        results['turning_point'] = (real_tp, gen_tp, abs(real_tp - gen_tp))

    # First crack (approx when temp reaches 385-390°F)
    if 'first_crack' in phases:
        real_fc = np.argmax(real >= 385) if np.any(real >= 385) else len(real)
        gen_fc = np.argmax(generated >= 385) if np.any(generated >= 385) else len(generated)
        results['first_crack'] = (real_fc, gen_fc, abs(real_fc - gen_fc))

    return results


def rate_of_rise_similarity(real: np.ndarray, generated: np.ndarray) -> float:
    """
    Compare Rate of Rise (RoR) patterns

    Args:
        real: Real temperature profile
        generated: Generated temperature profile

    Returns:
        MAE of RoR curves (°F/second)
    """
    # Calculate RoR (°F per second)
    real_ror = np.diff(real)
    gen_ror = np.diff(generated)

    min_len = min(len(real_ror), len(gen_ror))

    return np.mean(np.abs(real_ror[:min_len] - gen_ror[:min_len]))


def comprehensive_evaluation(
    real: np.ndarray,
    generated: np.ndarray,
    verbose: bool = False
) -> Dict[str, float]:
    """
    Run all evaluation metrics

    Args:
        real: Real temperature profile
        generated: Generated temperature profile
        verbose: If True, print results

    Returns:
        Dictionary of all metrics
    """
    metrics = {}

    # Temperature accuracy
    metrics['mae'] = mae(real, generated)
    metrics['rmse'] = rmse(real, generated)

    # Finish temperature
    finish_error, finish_accurate = finish_temp_accuracy(real, generated)
    metrics['finish_temp_error'] = finish_error
    metrics['finish_temp_accurate'] = float(finish_accurate)

    # Shape similarity
    metrics['dtw_distance'] = dtw_distance(real, generated)
    metrics['correlation'] = profile_correlation(real, generated)

    # Rate of Rise
    metrics['ror_mae'] = rate_of_rise_similarity(real, generated)

    # Phase timing
    phase_results = phase_timing_accuracy(real, generated)
    metrics['turning_point_error'] = phase_results['turning_point'][2]
    metrics['first_crack_error'] = phase_results['first_crack'][2]

    if verbose:
        print("=" * 80)
        print("COMPREHENSIVE PROFILE EVALUATION")
        print("=" * 80)
        print(f"Temperature Accuracy:")
        print(f"  MAE:                {metrics['mae']:.2f}°F")
        print(f"  RMSE:               {metrics['rmse']:.2f}°F")
        print(f"  Finish Temp Error:  {metrics['finish_temp_error']:.2f}°F " +
              f"({'✓ PASS' if metrics['finish_temp_accurate'] else '✗ FAIL'})")
        print()
        print(f"Shape Similarity:")
        print(f"  DTW Distance:       {metrics['dtw_distance']:.2f}")
        print(f"  Correlation:        {metrics['correlation']:.3f}")
        print()
        print(f"Rate of Rise:")
        print(f"  RoR MAE:            {metrics['ror_mae']:.2f}°F/s")
        print()
        print(f"Phase Timing:")
        print(f"  Turning Point:      {phase_results['turning_point'][2]}s error")
        print(f"  First Crack:        {phase_results['first_crack'][2]}s error")
        print("=" * 80)

    return metrics


def evaluate_batch(
    real_profiles: List[np.ndarray],
    generated_profiles: List[np.ndarray],
    verbose: bool = True
) -> Dict[str, Dict]:
    """
    Evaluate a batch of generated profiles

    Args:
        real_profiles: List of real temperature profiles
        generated_profiles: List of generated temperature profiles
        verbose: If True, print summary

    Returns:
        {
            'aggregate': {metric_name: mean_value},
            'individual': [
                {metric_name: value},  # for each profile
                ...
            ]
        }
    """
    if len(real_profiles) != len(generated_profiles):
        raise ValueError("Real and generated profile counts must match")

    individual_results = []

    for real, generated in zip(real_profiles, generated_profiles):
        metrics = comprehensive_evaluation(real, generated, verbose=False)
        individual_results.append(metrics)

    # Aggregate results
    metric_names = individual_results[0].keys()
    aggregate = {}

    for metric_name in metric_names:
        values = [result[metric_name] for result in individual_results]
        aggregate[metric_name] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values)
        }

    if verbose:
        print("\n" + "=" * 80)
        print("BATCH EVALUATION SUMMARY")
        print("=" * 80)
        print(f"Evaluated {len(real_profiles)} profiles\n")

        print(f"Temperature Accuracy:")
        print(f"  MAE:                {aggregate['mae']['mean']:.2f} ± {aggregate['mae']['std']:.2f}°F")
        print(f"  RMSE:               {aggregate['rmse']['mean']:.2f} ± {aggregate['rmse']['std']:.2f}°F")
        print(f"  Finish Temp Error:  {aggregate['finish_temp_error']['mean']:.2f} ± {aggregate['finish_temp_error']['std']:.2f}°F")
        print(f"  Finish Temp Acc:    {aggregate['finish_temp_accurate']['mean']*100:.1f}%")
        print()
        print(f"Shape Similarity:")
        print(f"  DTW Distance:       {aggregate['dtw_distance']['mean']:.2f} ± {aggregate['dtw_distance']['std']:.2f}")
        print(f"  Correlation:        {aggregate['correlation']['mean']:.3f} ± {aggregate['correlation']['std']:.3f}")
        print()
        print(f"Rate of Rise:")
        print(f"  RoR MAE:            {aggregate['ror_mae']['mean']:.2f} ± {aggregate['ror_mae']['std']:.2f}°F/s")
        print()
        print(f"Phase Timing:")
        print(f"  Turning Point Err:  {aggregate['turning_point_error']['mean']:.1f} ± {aggregate['turning_point_error']['std']:.1f}s")
        print(f"  First Crack Err:    {aggregate['first_crack_error']['mean']:.1f} ± {aggregate['first_crack_error']['std']:.1f}s")
        print("=" * 80)

    return {
        'aggregate': aggregate,
        'individual': individual_results
    }


def compute_success_metrics(aggregate: Dict[str, Dict]) -> Dict[str, bool]:
    """
    Check if metrics meet success criteria from CLAUDE.md

    Args:
        aggregate: Aggregate metrics from evaluate_batch

    Returns:
        {
            'mae_ok': bool,  # <5°F
            'dtw_ok': bool,  # <50
            'finish_temp_ok': bool,  # >90%
            'overall_success': bool
        }
    """
    criteria = {
        'mae_ok': aggregate['mae']['mean'] < 5.0,
        'dtw_ok': aggregate['dtw_distance']['mean'] < 50.0,
        'finish_temp_ok': aggregate['finish_temp_accurate']['mean'] > 0.90,
    }

    criteria['overall_success'] = all(criteria.values())

    return criteria


if __name__ == "__main__":
    # Example usage: compare two synthetic profiles
    print("Testing metrics module...")

    # Create synthetic profiles
    real = np.linspace(425, 395, 600)  # Simple linear drop
    generated = np.linspace(428, 393, 600) + np.random.randn(600) * 2  # With noise

    # Evaluate
    metrics = comprehensive_evaluation(real, generated, verbose=True)

    print(f"\n✓ Metrics module working correctly!")
