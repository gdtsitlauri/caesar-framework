"""
CAESAR — Metrics & Evaluation
==============================
PhD-level evaluation suite:
  • robustness_score        – defender's resistance to novel attacks
  • generalization_score    – performance on unseen attack types
  • detection_rate          – true positive rate
  • false_positive_rate     – FPR
  • neutralization_rate     – fraction of attacks fully stopped (success < 0.10)
  • co_evolutionary_gap     – F_def − F_att (positive = defender winning)
  • convergence_episode     – when defender fitness stabilises
"""

from __future__ import annotations

from typing import Dict, List, Optional
import numpy as np


# ═══════════════════════════════════════════════════════════════════════
def compute_episode_metrics(logs: List[Dict]) -> Dict:
    """Aggregate metrics over all training episodes."""
    if not logs:
        return {}

    def _arr(key):
        return np.array([m[key] for m in logs if key in m], dtype=np.float32)

    dr  = _arr('avg_defense_reward')
    asr = _arr('avg_attack_success')
    det = _arr('avg_detection_rate')
    fpr = _arr('avg_fpr')
    hlt = _arr('network_health')
    f_a = _arr('attacker_fitness')
    f_d = _arr('defender_fitness')
    pro = _arr('n_proactive')

    return {
        # Core metrics
        'mean_defense_reward':     float(dr.mean()),
        'mean_attack_success':     float(asr.mean()),
        'mean_detection_rate':     float(det.mean()),
        'mean_fpr':                float(fpr.mean()),
        'mean_network_health':     float(hlt.mean()),

        # Terminal (last 10 episodes)
        'final_defense_reward':    float(dr[-10:].mean()) if len(dr) >= 10 else float(dr.mean()),
        'final_attack_success':    float(asr[-10:].mean()) if len(asr) >= 10 else float(asr.mean()),
        'final_network_health':    float(hlt[-10:].mean()) if len(hlt) >= 10 else float(hlt.mean()),

        # Fitness
        'attacker_fitness_mean':   float(f_a.mean()),
        'defender_fitness_mean':   float(f_d.mean()),
        'co_evolutionary_gap':     float((f_d - f_a).mean()),

        # Novel
        'robustness_score':        _robustness(dr),
        'neutralization_rate':     _neutralization(asr),
        'convergence_episode':     _convergence(f_d),
        'total_proactive':         int(pro.sum()),
    }


def _robustness(defense_rewards: np.ndarray, window: int = 10) -> float:
    """
    Robustness = 1 − (std of trailing defense rewards).
    Higher = more stable/robust defense.
    """
    if len(defense_rewards) < window:
        return 0.0
    tail = defense_rewards[-window:]
    return float(max(0.0, 1.0 - tail.std()))


def _neutralization(attack_success: np.ndarray, threshold: float = 0.10) -> float:
    """Fraction of steps where attack success < threshold (fully neutralised)."""
    if len(attack_success) == 0:
        return 0.0
    return float((attack_success < threshold).mean())


def _convergence(defender_fitness: np.ndarray,
                 window: int = 15,
                 tol:    float = 0.02) -> int:
    """
    First episode where the defender fitness rolling std drops below tol.
    Returns -1 if not yet converged.
    """
    for i in range(window, len(defender_fitness)):
        if defender_fitness[i - window:i].std() < tol:
            return i
    return -1


# ═══════════════════════════════════════════════════════════════════════
def print_report(metrics: Dict, title: str = "CAESAR Evaluation Report") -> None:
    """Pretty-print the evaluation report."""
    width = 60
    print("\n" + "═" * width)
    print(f"  {title}")
    print("═" * width)

    sections = {
        "Core Performance": [
            'mean_defense_reward', 'mean_attack_success',
            'mean_detection_rate', 'mean_fpr', 'mean_network_health',
        ],
        "Terminal (Last 10 Episodes)": [
            'final_defense_reward', 'final_attack_success', 'final_network_health',
        ],
        "Co-evolutionary Fitness": [
            'attacker_fitness_mean', 'defender_fitness_mean', 'co_evolutionary_gap',
        ],
        "Novel CAESAR Metrics": [
            'robustness_score', 'neutralization_rate',
            'convergence_episode', 'total_proactive',
        ],
    }

    for section, keys in sections.items():
        print(f"\n  ▶ {section}")
        for k in keys:
            if k in metrics:
                v = metrics[k]
                if isinstance(v, float):
                    print(f"    {k:<32s}: {v:.4f}")
                else:
                    print(f"    {k:<32s}: {v}")

    print("\n" + "═" * width + "\n")


# ═══════════════════════════════════════════════════════════════════════
def per_attack_breakdown(logs: List[Dict],
                         env_attack_log: List[Dict]) -> Dict:
    """
    Compute per-attack-type performance breakdown.
    Returns dict keyed by attack type index.
    """
    from collections import defaultdict
    success_by_type: dict = defaultdict(list)
    for rec in env_attack_log:
        success_by_type[int(rec['type'])].append(float(rec['success']))

    result = {}
    for atype, successes in success_by_type.items():
        arr = np.array(successes, dtype=np.float32)
        result[atype] = {
            'n_attacks':         len(arr),
            'mean_success':      float(arr.mean()),
            'max_success':       float(arr.max()),
            'neutralization':    float((arr < 0.10).mean()),
        }
    return result
