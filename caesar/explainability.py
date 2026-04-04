"""
CAESAR Phase 2 — Explainability Module
========================================
PhD-level contribution: making the ADPN *explainable*.

Implements two complementary XAI techniques:
  1. Feature Importance (permutation-based) — which input features
     most affect the defender's action selection
  2. Q-value Attribution — which state dimensions pull the defender
     toward specific countermeasures (gradient-free, numpy-compatible)
  3. Defense Decision Map — visualises which attack patterns trigger
     which countermeasures (rule extraction from Q-function)

These visualisations satisfy the "explainability" requirement of the
PhD and are directly publishable as XAI-in-security contributions.
"""

from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import List, Dict, Optional

from environment import ATTACK_NAMES, DEFENSE_NAMES


# ═══════════════════════════════════════════════════════════════════════
def permutation_importance(
        adpn,
        states:   np.ndarray,
        feature_names: Optional[List[str]] = None,
        n_repeats: int = 10,
) -> Dict[str, np.ndarray]:
    """
    Permutation-based feature importance for the ADPN.

    For each feature, randomly shuffle its values across all samples
    and measure the drop in Q-value magnitude — the larger the drop,
    the more important the feature.
    """
    n, d = states.shape
    if feature_names is None:
        feature_names = [f'F{i:02d}' for i in range(d)]

    # Baseline: mean max Q-value on unperturbed states
    baseline_q = np.array([adpn.q_values(s).max() for s in states])
    baseline   = baseline_q.mean()

    importances = np.zeros(d)
    for fi in range(d):
        drop_sum = 0.0
        for _ in range(n_repeats):
            perturbed = states.copy()
            perturbed[:, fi] = np.random.permutation(perturbed[:, fi])
            perturbed_q = np.array([adpn.q_values(s).max() for s in perturbed])
            drop_sum += baseline - perturbed_q.mean()
        importances[fi] = drop_sum / n_repeats

    # Normalise to [0, 1]
    importances = np.clip(importances, 0, None)
    if importances.max() > 0:
        importances /= importances.max()

    idx = np.argsort(importances)[::-1]
    return {
        'importances':    importances,
        'feature_names':  np.array(feature_names),
        'sorted_idx':     idx,
        'baseline_q':     baseline,
    }


# ═══════════════════════════════════════════════════════════════════════
def q_attribution_heatmap(
        adpn,
        n_samples: int = 200,
        seed:      int = 42,
) -> np.ndarray:
    """
    Build a (n_defenses × n_features) attribution matrix.

    For each defense action j and each state feature i,
    measure: E[Q_j(s)] when feature i is high vs low.
    Large difference → feature i is decision-critical for action j.
    """
    rng = np.random.default_rng(seed)
    n_def     = adpn.n_actions
    state_dim = adpn.state_dim

    # Random state samples
    states = rng.uniform(0, 1, (n_samples, state_dim)).astype(np.float32)

    attr = np.zeros((n_def, state_dim), dtype=np.float32)
    for fi in range(state_dim):
        # High-value states (top quartile)
        hi_mask = states[:, fi] > np.percentile(states[:, fi], 75)
        lo_mask = states[:, fi] < np.percentile(states[:, fi], 25)

        q_hi = np.array([adpn.q_values(states[i]) for i in np.where(hi_mask)[0]])
        q_lo = np.array([adpn.q_values(states[i]) for i in np.where(lo_mask)[0]])

        if len(q_hi) > 0 and len(q_lo) > 0:
            attr[:, fi] = q_hi.mean(axis=0) - q_lo.mean(axis=0)

    return attr


# ═══════════════════════════════════════════════════════════════════════
def extract_decision_rules(
        adpn,
        env,
        n_probes: int = 500,
        seed:     int = 42,
) -> pd.DataFrame:
    """
    Probe the ADPN across attack types and intensities;
    extract which defense action it selects most often → rule table.
    """
    import pandas as pd
    rng = np.random.default_rng(seed)
    records = []

    for atk_type in range(env.N_ATTACKS):
        for intensity_bin in [0.2, 0.5, 0.8]:
            env.reset()
            env.inject(atk_type, intensity_bin)
            state = env._vec()
            q     = adpn.q_values(state)
            action = int(q.argmax())
            records.append({
                'attack_type':   ATTACK_NAMES[atk_type],
                'intensity_bin': f'{intensity_bin:.1f}',
                'chosen_action': DEFENSE_NAMES[action],
                'q_max':         float(q.max()),
                'q_diff':        float(q.max() - np.sort(q)[-2]),
            })

    return pd.DataFrame(records)


# ═══════════════════════════════════════════════════════════════════════
# Visualisations
# ═══════════════════════════════════════════════════════════════════════

PAL = dict(
    bg='#0d1117', panel='#161b22', txt='#c9d1d9', grid='#21262d',
    atk='#e74c3c', df='#2980b9', hi='#f39c12', pos='#27ae60', neg='#e74c3c',
)


def _sa(ax, title='', xlabel='', ylabel=''):
    ax.set_facecolor(PAL['panel'])
    ax.tick_params(colors=PAL['txt'])
    for lab in [ax.xaxis.label, ax.yaxis.label, ax.title]:
        lab.set_color(PAL['txt'])
    for sp in ax.spines.values():
        sp.set_color(PAL['grid'])
    ax.grid(True, color=PAL['grid'], lw=.6, alpha=.7)
    if title:  ax.set_title(title,  fontsize=10, fontweight='bold', color=PAL['txt'])
    if xlabel: ax.set_xlabel(xlabel, fontsize=8)
    if ylabel: ax.set_ylabel(ylabel, fontsize=8)


def plot_feature_importance(
        importance_dict: Dict,
        top_k:     int  = 16,
        save_path: str  = 'results/explainability_importance.png',
) -> str:
    idx   = importance_dict['sorted_idx'][:top_k]
    names = importance_dict['feature_names'][idx]
    vals  = importance_dict['importances'][idx]

    fig, ax = plt.subplots(figsize=(11, 6), facecolor=PAL['bg'])
    colors  = [PAL['df'] if v > 0.5 else PAL['hi'] for v in vals]
    bars    = ax.barh(range(top_k), vals[::-1], color=colors[::-1],
                      alpha=.82, edgecolor='white', lw=.4)
    ax.set_yticks(range(top_k))
    ax.set_yticklabels(names[::-1], fontsize=8, color=PAL['txt'])
    _sa(ax, 'ADPN Feature Importance (Permutation-based)', 'Importance Score', '')
    ax.axvline(0.5, color='white', ls=':', lw=1, alpha=.6, label='threshold 0.5')
    ax.legend(fontsize=8, facecolor=PAL['panel'], labelcolor=PAL['txt'])
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=PAL['bg'])
    plt.close(fig)
    return save_path


def plot_q_attribution(
        attr:      np.ndarray,
        top_feat:  int = 16,
        save_path: str = 'results/explainability_qattr.png',
) -> str:
    n_def, n_feat = attr.shape
    show = min(top_feat, n_feat)
    # Pick most varying features
    var_idx = np.argsort(np.abs(attr).sum(axis=0))[::-1][:show]
    A = attr[:, var_idx]

    def_labels  = [d.replace('_', '\n') for d in DEFENSE_NAMES[:n_def]]
    feat_labels = [f'F{i:02d}' for i in var_idx]

    fig, ax = plt.subplots(figsize=(14, 5), facecolor=PAL['bg'])
    vmax = float(np.abs(A).max()) or 1.0
    im   = ax.imshow(A, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='auto')
    ax.set_xticks(range(show)); ax.set_yticks(range(n_def))
    ax.set_xticklabels(feat_labels, rotation=45, ha='right',
                       fontsize=7, color=PAL['txt'])
    ax.set_yticklabels(def_labels, fontsize=8, color=PAL['txt'])
    ax.set_title('Q-value Attribution Heatmap\n(red=action↑ when feature↑, blue=action↓)',
                 color=PAL['txt'], fontsize=11, fontweight='bold')
    ax.set_facecolor(PAL['panel'])
    plt.colorbar(im, ax=ax, label='Attribution (ΔQ)')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=PAL['bg'])
    plt.close(fig)
    return save_path


def plot_decision_rules(
        rules_df,
        save_path: str = 'results/explainability_rules.png',
) -> str:
    """Visualise the rule extraction table as a styled heatmap."""
    import pandas as pd

    pivot = rules_df.pivot_table(
        index='attack_type', columns='intensity_bin',
        values='chosen_action', aggfunc='first',
    )

    # Encode to integers for colour mapping
    all_actions = DEFENSE_NAMES
    enc = pivot.map(lambda x: all_actions.index(x) if x in all_actions else 0)

    fig, ax = plt.subplots(figsize=(10, 6), facecolor=PAL['bg'])
    im = ax.imshow(enc.values, cmap='tab10', aspect='auto',
                   vmin=0, vmax=len(all_actions) - 1)

    ax.set_xticks(range(enc.shape[1]))
    ax.set_yticks(range(enc.shape[0]))
    ax.set_xticklabels([f'Intensity {c}' for c in enc.columns],
                       fontsize=9, color=PAL['txt'])
    ax.set_yticklabels(enc.index, fontsize=8, color=PAL['txt'])
    ax.set_title('ADPN Decision Rules: Attack × Intensity → Defense',
                 color=PAL['txt'], fontsize=11, fontweight='bold')
    ax.set_facecolor(PAL['panel'])

    # Annotate cells with action name
    for i in range(enc.shape[0]):
        for j in range(enc.shape[1]):
            txt = pivot.values[i, j]
            if txt:
                ax.text(j, i, txt.replace('_', '\n'),
                        ha='center', va='center', fontsize=6.5,
                        color='white', fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=PAL['bg'])
    plt.close(fig)
    return save_path


# ── import guard
try:
    import pandas as pd
except ImportError:
    pass
